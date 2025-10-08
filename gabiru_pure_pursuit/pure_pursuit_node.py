import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Twist
from std_srvs.srv import Trigger
import numpy as np
import csv
import os

class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        # Parámetros por defecto (se sobrescriben con el CSV)
        self.declare_parameter('wheelbase', 0.16)  # TurtleBot3 Burger wheelbase
        self.wheelbase = self.get_parameter('wheelbase').value
        # Suscripciones

        #Nota: En el programa en general solo falta que esta Odometry funcione en conjunto con Gazebo,rviz, nav2 etc
        #Como esta el programa ahora mismo, se publica el primer segmento y, como Odometry no cambia, nos quedamos publicando este segmento.
        self.subscription_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.subscription_path = self.create_subscription(PoseArray, '/optimal_path', self.path_callback, 10)
        # Publicador
        self.publisher_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        # Servicio para sincronización
        self.srv = self.create_service(Trigger, 'ready_to_pursue', self.ready_callback)
        self.srv_online = self.create_service(Trigger,'ready_to_receive_path',self.ready_callback)#Para poderme comunicar con el best_path en el online
        # Estado
        self.current_pos = np.array([0.0, 0.0])  # [x, y]
        self.current_theta = 0.0
        self.waypoints = None  # Array de [x, y]
        self.segments_params = {}  # Dict de params por segment_id
        self.current_segment_id = 0
        # Cargar parámetros del CSV
        self.csv_path = os.path.expanduser("~/sim_ws/optimized_PP_params.csv")
        self.load_params_from_csv()
        # Timer para control
        self.timer = self.create_timer(0.05, self.control_callback)  # 20 Hz
        self.get_logger().info("PurePursuitNode iniciado, esperando /optimal_path...")

    def load_params_from_csv(self):
        """Carga parámetros optimizados desde el CSV."""
        if not os.path.isfile(self.csv_path):
            self.get_logger().error(f"CSV no encontrado en {self.csv_path}")
            return
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                seg_id = int(row['segment_id'])
                self.segments_params[seg_id] = {
                    'Ld': float(row['Ld']),
                    'vd': float(row['vd']),
                    'w_max': float(row['w_max']),
                    'start_idx': int(row['start_idx']),
                    'end_idx': int(row['end_idx'])
                }
        self.get_logger().info(f"Cargados params para {len(self.segments_params)} tramos")

    def ready_callback(self, request, response):
        """Responde al servicio ready_to_pursue."""
        response.success = True
        response.message = "PurePursuit listo para recibir waypoints"
        self.get_logger().info("Servicio ready_to_pursue activado")
        return response

    def odom_callback(self, msg):
        """Actualiza la posición y orientación del vehículo desde /odom."""
        self.current_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.current_theta = 2 * np.arctan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)  # Quaternion a yaw

    def path_callback(self, msg):
        """Almacena waypoints de /optimal_path."""
        self.waypoints = np.array([[pose.position.x, pose.position.y] for pose in msg.poses])
        self.get_logger().info(f"Recibidos {len(self.waypoints)} waypoints en /optimal_path")

    def control_callback(self):
        """Implementa Pure Pursuit y publica /cmd_vel."""
        if self.waypoints is None or not self.segments_params:
            return
        # Encontrar waypoint más cercano
        distances = np.linalg.norm(self.waypoints - self.current_pos, axis=1)
        nearest_idx = np.argmin(distances)
        # Determinar tramo actual basado en índice
        for seg_id, params in self.segments_params.items():
            if params['start_idx'] <= nearest_idx <= params['end_idx']:
                self.current_segment_id = seg_id
                break
        else:
            self.get_logger().warn(f"No se encontró tramo para índice {nearest_idx}")
            return
        # Obtener parámetros del tramo actual
        params = self.segments_params[self.current_segment_id]
        Ld = params['Ld']
        vd = min(params['vd'], 0.22)  # Limitar para TurtleBot3 Burger
        w_max = min(params['w_max'], 2.84)  # Limitar para TurtleBot3 Burger
        # Calcular look-ahead point
        look_ahead_idx = nearest_idx
        cum_dist = 0.0
        while look_ahead_idx < len(self.waypoints) - 1 and cum_dist < Ld:
            cum_dist += np.linalg.norm(self.waypoints[look_ahead_idx + 1] - self.waypoints[look_ahead_idx])
            look_ahead_idx += 1
        if look_ahead_idx >= len(self.waypoints):
            look_ahead_idx = len(self.waypoints) - 1
        goal_point = self.waypoints[look_ahead_idx]
        # Calcular ángulo relativo (alpha)
        alpha = np.arctan2(goal_point[1] - self.current_pos[1], goal_point[0] - self.current_pos[0]) - self.current_theta
        # Calcular curvatura y velocidad angular
        kappa = 2 * np.sin(alpha) / Ld
        w = vd * kappa
        # Limitar velocidad angular
        if abs(w) > w_max:
            w = np.sign(w) * w_max
        # Publicar comando
        cmd = Twist()
        cmd.linear.x = vd
        cmd.angular.z = w
        self.publisher_cmd.publish(cmd)
        self.get_logger().info(f"Tramo {self.current_segment_id}: Ld={Ld:.2f}, vd={vd:.2f}, w={w:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()