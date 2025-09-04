import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from gabiru_shared.csv_utils import load_csv, get_shared_data_path
from geometry_msgs.msg import Twist
import numpy as np
import json
import math
import os
import csv
import time

class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # Parámetros del nodo
        self.declare_parameter('wheelbase', 0.3)  # Distancia entre ejes del vehículo (en metros)
        self.wheelbase = self.get_parameter('wheelbase').value
        self.csv_params_path = get_shared_data_path('optimized_PP_parms.csv')
        self.csv_commands_path = get_shared_data_path('control_commands.csv')

        # QoS para suscripciones y publicaciones
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_ALL,
            depth=1000
        )

        # Suscripciones
        self.segments_sub = self.create_subscription(
            String,
            'gabiru/segments',
            self.segments_callback,
            qos)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos)

        # Publicación de comandos de control
        self.cmd_pub = self.create_publisher(String, 'gabiru/control_commands', qos)

        #Publicacion de los comandos de velocidad para RViz
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', qos)

        # Almacenamiento de datos
        self.segments = {}  # Diccionario para almacenar waypoints por segment_id
        self.params = {}    # Diccionario para almacenar parámetros optimizados por segment_id
        self.current_pose = None  # Posición actual del vehículo
        self.current_segment_id = 0  # Segmento actual
        self.all_segments_optimized = False  # Bandera para verificar optimización completa

        # Temporizador para verificar parámetros optimizados
        self.timer = self.create_timer(1.0, self.check_optimization_status)

    def check_optimization_status(self):
        """Verifica si todos los segmentos están optimizados en el CSV."""
        if self.all_segments_optimized:
            return

        if not os.path.exists(self.csv_params_path):
            self.get_logger().info("Esperando archivo CSV con parámetros optimizados...")
            return

        try:
            df = load_csv('optimized_PP_parms.csv')
            optimized_segment_ids = set(df['segment_id'].values)
            received_segment_ids = set(self.segments.keys())

            if received_segment_ids and optimized_segment_ids == received_segment_ids:
                self.get_logger().info("Todos los segmentos están optimizados. Iniciando Pure Pursuit.")
                self.all_segments_optimized = True
                # Cargar parámetros desde CSV
                for _, row in df.iterrows():
                    segment_id = int(row['segment_id'])
                    self.params[segment_id] = {
                        'Ld': float(row['Ld']),
                        'vd': float(row['vd']),
                        'w_max': float(row['w_max'])
                    }
            else:
                self.get_logger().info(f"Esperando optimización completa. Recibidos: {len(received_segment_ids)}, Optimizados: {len(optimized_segment_ids)}")
        except Exception as e:
            self.get_logger().error(f"Error al leer CSV: {e}")

    def segments_callback(self, msg):
        """Callback para almacenar los waypoints de los segmentos."""
        try:
            data = json.loads(msg.data)
            segment_id = data["segment_id"]
            waypoints = np.array(data["waypoints"])  # Convertir a numpy array
            self.segments[segment_id] = {
                "tipo": data["tipo"],
                "waypoints": waypoints
            }
            self.get_logger().info(f"Recibido segmento {segment_id} con {len(waypoints)} waypoints")
        except Exception as e:
            self.get_logger().error(f"Error al parsear segmento: {e}")

    def odom_callback(self, msg):
        """Callback para actualizar la posición actual del vehículo."""
        self.current_pose = msg.pose.pose
        if self.all_segments_optimized:
            self.compute_pure_pursuit()

    def save_control_commands(self, segment_id, v, w):
        """Guarda los comandos de control en un CSV."""
        file_exists = os.path.isfile(self.csv_commands_path)
        with open(self.csv_commands_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["timestamp", "segment_id", "v", "w"])
            writer.writerow([time.time(), segment_id, v, w])

    def compute_pure_pursuit(self):
        """Calcula el ángulo de giro y la velocidad usando Pure Pursuit."""
        if self.current_pose is None or not self.segments or not self.params:
            self.get_logger().warn("Datos insuficientes (pose, segmentos o parámetros no disponibles)")
            return

        # Obtener waypoints y parámetros del segmento actual
        if self.current_segment_id not in self.segments or self.current_segment_id not in self.params:
            self.get_logger().warn(f"Segmento {self.current_segment_id} no disponible")
            return

        waypoints = self.segments[self.current_segment_id]["waypoints"]
        params = self.params[self.current_segment_id]
        Ld = params["Ld"]  # Lookahead distance
        vd = params["vd"]  # Velocidad deseada
        w_max = params["w_max"]  # Velocidad angular máxima

        # Posición y orientación actual del vehículo
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        quaternion = (
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w
        )
        # Convertir quaternion a yaw (en radianes)
        yaw = self.quaternion_to_yaw(quaternion)

        # Encontrar el waypoint objetivo (el más cercano dentro de Ld)
        target_wp = self.find_target_waypoint(waypoints, x, y, Ld)

        if target_wp is None:
            self.get_logger().warn("No se encontró un waypoint objetivo")
            return

        # Calcular el ángulo de giro usando Pure Pursuit
        steering_angle = self.calculate_steering_angle(x, y, yaw, target_wp, Ld)

        # Limitar la velocidad angular
        steering_angle = np.clip(steering_angle, -w_max, w_max)

        # Publicar comando de control
        cmd = String()
        cmd_data = {
            "timestamp": time.time(),
            "segment_id": self.current_segment_id,
            "v": vd,
            "w": steering_angle
        }
        cmd.data = json.dumps(cmd_data)
        self.cmd_pub.publish(cmd)
        self.get_logger().info(f"Comando publicado para segmento {self.current_segment_id}: {cmd.data}")


        #Publicar velocidad y direccion para RViz       
        cmd = Twist()
        cmd.linear.x = vd  # Velocidad lineal en el eje x
        cmd.angular.z = steering_angle  # Velocidad angular en el eje z (yaw)
        self.cmd_pub.publish(cmd)
        self.get_logger().info(f"Comando publicado para segmento {self.current_segment_id}: v={vd}, w={steering_angle}")

        # Guardar comando en CSV
        self.save_control_commands(self.current_segment_id, vd, steering_angle)

    def find_target_waypoint(self, waypoints, x, y, Ld):
        """Encuentra el waypoint objetivo más cercano dentro de la distancia Ld."""
        min_dist = float('inf')
        target_wp = None

        for wp in waypoints:
            dist = np.sqrt((wp[0] - x)**2 + (wp[1] - y)**2)
            if dist <= Ld and dist < min_dist:
                min_dist = dist
                target_wp = wp

        return target_wp

    def calculate_steering_angle(self, x, y, yaw, target_wp, Ld):
        """Calcula el ángulo de giro usando el algoritmo Pure Pursuit."""
        # Coordenadas del waypoint objetivo
        target_x, target_y = target_wp[0], target_wp[1]

        # Transformar el waypoint objetivo al sistema de coordenadas del vehículo
        dx = target_x - x
        dy = target_y - y
        alpha = math.atan2(dy, dx) - yaw
        alpha = self.normalize_angle(alpha)

        # Calcular el ángulo de giro
        steering_angle = math.atan2(2.0 * self.wheelbase * math.sin(alpha), Ld)
        return steering_angle

    def normalize_angle(self, angle):
        """Normaliza un ángulo al rango [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def quaternion_to_yaw(self, quaternion):
        """Convierte un quaternion a ángulo yaw."""
        x, y, z, w = quaternion
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    node.get_logger().info("Nodo Pure Pursuit iniciado")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()