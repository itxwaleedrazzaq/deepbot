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
    maintainer='Waleed Razzaq',
    maintainer_email='waleedrazzaq@mail.ustc.edu.cn',
    description='This package contains the test nodes for deepbot',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'test_ai_node = deepbot_test.test_ai:main',
        ],
    },
)
