from setuptools import setup
from glob import glob
import os

package_name = 'arm_vla_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, f'{package_name}.nodes'],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='VLA pipeline for 5-DOF robotic arm',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_node = arm_vla_pkg.nodes.vision_node:main',
            'speech_node = arm_vla_pkg.nodes.speech_node:main',
            'brain_node = arm_vla_pkg.nodes.brain_node:main',
            'action_node = arm_vla_pkg.nodes.action_node:main',
            'visual_servoing_node = arm_vla_pkg.nodes.visual_servoing_node:main',
            'test_visual_servoing = arm_vla_pkg.nodes.test_visual_servoing:main',
            'visual_servoing_calibration = arm_vla_pkg.nodes.visual_servoing_calibration:main',
            'way_point = arm_vla_pkg.nodes.way_point:main',

        ],
    },
)
