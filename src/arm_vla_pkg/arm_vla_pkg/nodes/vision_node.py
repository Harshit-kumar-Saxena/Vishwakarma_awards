#!/usr/bin/env python3
"""
Vision Node - Detection JSON Publisher
Subscribes to TF transforms published by publish_tf.py
Converts them to JSON format for brain_node
Publishes to /detections
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import TransformListener, Buffer
import json
import math


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Publisher
        self.detections_pub = self.create_publisher(String, '/detections', 10)
        
        # Object names to track (adjust based on your setup)
        self.tracked_objects = [
            'ball_0', 'ball_1', 'ball_2', 'ball_3',
            'pink_plate', 'bowl'
        ]
        
        self.reference_frame = 'base_link'
        
        # Timer to publish detections
        self.create_timer(0.5, self.publish_detections)  # 2 Hz
        
        self.get_logger().info('👁️  Vision Node initialized. Tracking objects...')

    def publish_detections(self):
        """Query TF and publish JSON detections"""
        detections = {}
        
        for obj_name in self.tracked_objects:
            try:
                # Get transform from base_link to object
                transform = self.tf_buffer.lookup_transform(
                    self.reference_frame,
                    obj_name,
                    rclpy.time.Time()
                )
                
                # Extract position
                pos = transform.transform.translation
                detections[obj_name] = {
                    'position': [
                        round(pos.x, 3),
                        round(pos.y, 3),
                        round(pos.z, 3)
                    ],
                    'frame': self.reference_frame
                }
                
            except Exception as e:
                # Object not detected or TF not available
                self.get_logger().debug(f'{obj_name} not found: {e}')
                continue
        
        # Publish JSON
        if detections:
            json_msg = String()
            json_msg.data = json.dumps(detections)
            self.detections_pub.publish(json_msg)
            
            self.get_logger().debug(f'Published {len(detections)} detections')
        else:
            self.get_logger().warn('No objects detected')


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
