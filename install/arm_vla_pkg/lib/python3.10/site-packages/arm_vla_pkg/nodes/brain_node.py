#!/usr/bin/env python3
"""
Brain Node - VLA Intelligence Layer
Receives: voice command + vision detections
Sends to: Ollama LLM
Outputs: structured action JSON
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import requests
import json
import re


class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')
        
        # Publishers & Subscribers
        self.action_pub = self.create_publisher(String, '/vla/action_command', 10)
        self.create_subscription(String, '/vla/voice_command', self.voice_callback, 10)
        self.create_subscription(String, '/detections', self.vision_callback, 10)
        
        # State
        self.latest_detections = None
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "mistral:7b"  # Faster model (was mistral:7b)
        
        self.get_logger().info('🧠 Brain Node initialized. Waiting for commands...')

    def vision_callback(self, msg):
        """Store latest vision detections"""
        try:
            self.latest_detections = json.loads(msg.data)
            self.get_logger().debug(f'Vision updated: {len(self.latest_detections)} objects detected')
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON from /detections')

    def voice_callback(self, msg):
        """Process voice command with Ollama"""
        command = msg.data.strip()
        self.get_logger().info(f'📢 Received command: "{command}"')
        
        if not self.latest_detections:
            self.get_logger().warn('⚠️  No vision data available yet!')
            return
        
        # Generate action from Ollama
        action_json = self.query_ollama(command, self.latest_detections)
        
        if action_json:
            self.get_logger().info(f'✅ Action generated: {action_json}')
            self.action_pub.publish(String(data=json.dumps(action_json)))
        else:
            self.get_logger().error('❌ Failed to generate valid action')

    def query_ollama(self, command, detections):
        """Send prompt to Ollama and parse JSON response"""
        prompt = self.build_prompt(command, detections)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            self.get_logger().info('🤖 Querying Ollama...')
            response = requests.post(self.ollama_url, json=payload, timeout=60)  # Increased from 30 to 60
            response.raise_for_status()
            
            result = response.json()
            raw_output = result.get('response', '').strip()
            
            self.get_logger().debug(f'Raw Ollama output: {raw_output}')
            
            # Parse JSON from response
            action_json = self.extract_json(raw_output)
            return action_json
            
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f'Ollama request failed: {e}')
            return None
        except Exception as e:
            self.get_logger().error(f'Unexpected error: {e}')
            return None

    def build_prompt(self, command, detections):
        """Construct system prompt for Ollama"""
        objects_str = json.dumps(detections, indent=2)
        
        prompt = f"""You are a robot arm controller. Your task is to interpret natural language commands and generate precise pick-and-place actions.

**Available Objects (from vision system):**
```json
{objects_str}
```

**User Command:**
"{command}"

**Your Task:**
1. Identify which object to pick from the command
2. Find the object's current position from the vision data
3. Determine the target place location
4. Output ONLY a valid JSON with this exact structure:

{{
  "action": "pick_and_place",
  "object_id": "exact_object_name_from_detections",
  "pick": [x, y, z],
  "place": [x, y, z]
}}

**Rules:**
- object_id MUST match exactly from the detections list (e.g., "ball_0", "ball_1", "pink_plate")
- pick coordinates MUST come from the detected object's position array
- place coordinates should be the target location:
  * If "on pink_plate" or "on plate" → use pink_plate position
  * If "on bowl" → use bowl position
  * Otherwise use specific coordinates
- All coordinates in meters [x, y, z]
- z value should stay as detected (don't modify height)
- Output ONLY the JSON, no markdown, no explanations, no code blocks

**Examples:**
Command: "Pick ball_0 and place it on pink_plate"
If detections show:
- ball_0: [0.25, 0.15, 0.015]
- pink_plate: [0.40, -0.20, 0.01]
Output:
{{
  "action": "pick_and_place",
  "object_id": "ball_0",
  "pick": [0.25, 0.15, 0.015],
  "place": [0.40, -0.20, 0.01]
}}

Command: "move ball 1 to the plate"
If detections show:
- ball_1: [0.30, 0.18, 0.015]
- pink_plate: [0.40, -0.20, 0.01]
Output:
{{
  "action": "pick_and_place",
  "object_id": "ball_1",
  "pick": [0.30, 0.18, 0.015],
  "place": [0.40, -0.20, 0.01]
}}

Now generate ONLY the JSON for the given command (no other text):"""
        
        return prompt

    def extract_json(self, text):
        """Extract JSON from Ollama response"""
        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON object
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        self.get_logger().error(f'Could not extract JSON from: {text}')
        return None


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()