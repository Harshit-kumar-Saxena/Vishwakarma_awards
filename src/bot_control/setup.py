from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'bot_control'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'opencv-python',
        'numpy',
        'scipy',
    ],
    zip_safe=True,
    maintainer='aditya',
    maintainer_email='aditya.arora.emails@gmail.com',
    description='bot control package with vision',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'bot_controller = bot_control.bot_controller:main',
            'interactive_control = bot_control.interactive_control:main',
            'vision_pick_place = bot_control.vision_pick_place:main',
            'scanning_vision_pick_place = bot_control.scanning_vision_pick_place:main',
            'test_ee_camera = bot_control.test_ee_camera:main',
            'simple_ball_detector = bot_control.simple_ball_detector:main',
            'publish_tf = bot_control.publish_tf:main',
            'dual_camera_picker = bot_control.dual_camera_picker:main',
        ],
    },
)