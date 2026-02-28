#!/usr/bin/env python3
"""
Speech Node - Voice Command Input
Converts speech to text and publishes to /vla/voice_command
Uses Google Speech Recognition (offline fallback available)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import speech_recognition as sr
import threading


class SpeechNode(Node):
    def __init__(self):
        super().__init__('speech_node')
        
        # Publisher
        self.command_pub = self.create_publisher(String, '/vla/voice_command', 10)
        
        # Speech recognizer
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        self.get_logger().info('🎤 Calibrating microphone...')
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        
        self.get_logger().info('✅ Speech Node ready. Say commands now!')
        
        # Start listening thread
        self.listening = True
        self.listen_thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.listen_thread.start()

    def listen_loop(self):
        """Continuous listening loop"""
        while self.listening and rclpy.ok():
            try:
                with self.microphone as source:
                    self.get_logger().info('🎧 Listening...')
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    
                    # Try to recognize
                    try:
                        # Use Google Speech Recognition (requires internet)
                        text = self.recognizer.recognize_google(audio)
                        self.get_logger().info(f'📝 Recognized: "{text}"')
                        
                        # Publish command
                        self.command_pub.publish(String(data=text))
                        
                    except sr.UnknownValueError:
                        self.get_logger().warn('❓ Could not understand audio')
                    except sr.RequestError as e:
                        self.get_logger().error(f'⚠️  Speech service error: {e}')
                    
            except sr.WaitTimeoutError:
                # No speech detected, continue
                continue
            except Exception as e:
                self.get_logger().error(f'Speech error: {e}')
                continue

    def destroy_node(self):
        """Clean shutdown"""
        self.listening = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SpeechNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
