cmake_minimum_required(VERSION 3.8)
project(gabiru_pure_pursuit)

# Encontrar dependencias
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(gabiru_shared REQUIRED)

# Instalar módulos Python
ament_python_install_package(${PROJECT_NAME})

# Instalar scripts ejecutables
install(PROGRAMS
  gabiru_pure_pursuit/pure_pursuit_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Instalar archivos adicionales (si es necesario, por ejemplo, launch files)
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Exportar información del paquete
ament_package()