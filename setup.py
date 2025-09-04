from setuptools import setup

package_name = 'gabiru_pure_pursuit'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='carlos',
    maintainer_email='carlos@todo.todo',
    description='Nodo Pure Pursuit para el proyecto Gabiru',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pure_pursuit_node = gabiru_pure_pursuit.pure_pursuit_node:main',
        ],
    },
)