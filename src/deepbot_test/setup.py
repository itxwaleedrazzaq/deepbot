from setuptools import find_packages, setup

package_name = 'deepbot_test'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='waleed',
    maintainer_email='waleed@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'test_ai_node = deepbot_test.test_ai:main',
            'visualize_ai_node = deepbot_test.gradcam_visualize:main',
            'trajectory_recorder_node = deepbot_test.track_traj:main',
            'all_trajectory_recorder_node = deepbot_test.track_traj2:main',
            'test_ai_node_seed = deepbot_test.test_ai_seed:main',
            'trajectory_recorder_node_seed = deepbot_test.track_traj_seed:main',
            'all_trajectory_recorder_node_seed = deepbot_test.track_traj2_seed:main',
            'test_ai_coeff_node = deepbot_test.test_ai_coeff:main',

        ],
    },
)
