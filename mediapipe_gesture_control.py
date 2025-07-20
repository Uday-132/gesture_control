#!/usr/bin/env python3
"""
MediaPipe Hand Gesture Control System
Detects open hand and fist gestures using MediaPipe and OpenCV
Controls ArrowUp/ArrowDown keys for game control
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import argparse
from pynput.keyboard import Key
from pynput import keyboard
from pynput.mouse import Button, Listener as MouseListener
from pynput import mouse
import threading
import math

class MediaPipeGestureController:
    def __init__(self, confidence_threshold=0.7, detection_confidence=0.7, tracking_confidence=0.5):
        """
        Initialize MediaPipe hand gesture controller
        
        Args:
            confidence_threshold (float): Minimum confidence for gesture classification
            detection_confidence (float): MediaPipe hand detection confidence
            tracking_confidence (float): MediaPipe hand tracking confidence
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Only detect one hand for simplicity
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture state tracking
        self.current_gesture = 'none'
        self.gesture_history = deque(maxlen=7)  # Smooth over 7 frames
        self.gesture_confidence = 0.0
        
        # Key press state
        self.arrow_up_pressed = False
        self.arrow_down_pressed = False
        self.arrow_left_pressed = False
        self.arrow_right_pressed = False
        self.keyboard_controller = keyboard.Controller()
        self.key_lock = threading.Lock()
        
        # Mouse control state
        self.mouse_controller = mouse.Controller()
        self.cursor_mode = False  # Toggle between game control and cursor control
        self.last_cursor_pos = (0, 0)
        self.cursor_smoothing = deque(maxlen=5)  # Smooth cursor movement
        self.is_pinching = False
        self.is_dragging_cursor = False
        self.pinch_threshold = 0.05  # Distance threshold for pinch detection
        self.scroll_accumulator = 0.0  # For smooth scrolling
        self.last_hand_y = 0  # For scroll gesture detection
        
        # FPS tracking
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()
        
        # Interactive steering wheel state
        self.manual_rotation = 0  # Manual rotation angle (-90 to +90 degrees)
        self.is_dragging = False
        self.last_mouse_angle = 0
        self.wheel_center_x = 0
        self.wheel_center_y = 0
        self.wheel_radius = 80
        
        # Hand landmarks indices for finger detection
        self.finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        self.finger_pips = [3, 6, 10, 14, 18]  # Finger PIP joints
        
        print("🖐️ MediaPipe Hand Gesture Controller initialized")
        print("🎮 GAME MODE:")
        print("🤟 Three fingers → Left Arrow")
        print("✌️ Two fingers → Right Arrow") 
        print("🖐️ Open hand → ArrowUp (boost)")
        print("✊ Fist → ArrowDown (brake)")
        print("🖱️ CURSOR MODE (Press 'c' to toggle):")
        print("👆 Index finger → Move cursor")
        print("🤏 Pinch (thumb + index) → Left click")
        print("✊ Fist → Right click")
        print("🖐️ Open hand up/down → Scroll")
        print("📹 Starting webcam...")    

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_finger_extended(self, landmarks, finger_idx):
        """
        Check if a finger is extended based on landmarks
        
        Args:
            landmarks: Hand landmarks from MediaPipe
            finger_idx: Finger index (0=thumb, 1=index, 2=middle, 3=ring, 4=pinky)
            
        Returns:
            bool: True if finger is extended
        """
        if finger_idx == 0:  # Thumb - special case
            # For thumb, check if tip is further from wrist than MCP joint
            thumb_tip = landmarks[4]
            thumb_mcp = landmarks[2]
            wrist = landmarks[0]
            
            # Calculate distances from wrist
            tip_dist = self.calculate_distance(thumb_tip, wrist)
            mcp_dist = self.calculate_distance(thumb_mcp, wrist)
            
            return tip_dist > mcp_dist
        else:
            # For other fingers, check if tip is above PIP joint
            tip_idx = self.finger_tips[finger_idx]
            pip_idx = self.finger_pips[finger_idx]
            
            return landmarks[tip_idx].y < landmarks[pip_idx].y
    
    def count_extended_fingers(self, landmarks):
        """
        Count number of extended fingers
        
        Args:
            landmarks: Hand landmarks from MediaPipe
            
        Returns:
            tuple: (extended_count, finger_states)
        """
        finger_states = []
        
        for i in range(5):
            extended = self.is_finger_extended(landmarks, i)
            finger_states.append(extended)
        
        extended_count = sum(finger_states)
        return extended_count, finger_states    

    def classify_gesture(self, landmarks):
        """
        Classify hand gesture based on landmarks
        
        Args:
            landmarks: Hand landmarks from MediaPipe
            
        Returns:
            tuple: (gesture_name, confidence)
        """
        if not landmarks:
            return 'none', 0.0
        
        # Count extended fingers
        extended_count, finger_states = self.count_extended_fingers(landmarks)
        
        # Calculate hand openness based on finger spread
        hand_openness = self.calculate_hand_openness(landmarks)
        
        # Gesture classification logic
        if extended_count >= 4:  # 4 or 5 fingers extended
            # Open hand - high confidence if fingers are spread
            confidence = 0.9 if hand_openness > 0.15 else 0.7
            return 'open_hand', confidence
        
        elif extended_count == 0:  # No fingers extended
            # Fist - high confidence if hand is compact
            confidence = 0.9 if hand_openness < 0.1 else 0.7
            return 'fist', confidence
        
        elif extended_count == 1:  # One finger extended
            # Check which finger is extended
            if finger_states[1]:  # Index finger only
                return 'one_finger', 0.9
            elif finger_states[0]:  # Thumb only
                return 'thumb_up', 0.8
            else:  # Other single finger
                return 'one_finger', 0.7
        
        elif extended_count == 2:  # Two fingers extended
            # Check for specific two-finger combinations
            if finger_states[1] and finger_states[2]:  # Index and middle
                return 'two_fingers', 0.9
            elif finger_states[0] and finger_states[1]:  # Thumb and index
                return 'two_fingers', 0.8
            else:  # Other two-finger combination
                return 'two_fingers', 0.7
        
        elif extended_count == 3:
            # Three fingers - check for specific combinations
            if finger_states[1] and finger_states[2] and finger_states[3]:  # Index, middle, ring
                return 'three_fingers', 0.9
            elif finger_states[0] and finger_states[1] and finger_states[2]:  # Thumb, index, middle
                return 'three_fingers', 0.8
            else:  # Other three-finger combination
                return 'three_fingers', 0.7
        
        else:
            # Unclear gesture
            return 'unclear', 0.3   
 
    def calculate_hand_openness(self, landmarks):
        """
        Calculate how "open" the hand is based on finger spread
        
        Args:
            landmarks: Hand landmarks from MediaPipe
            
        Returns:
            float: Hand openness measure (0 = closed, 1 = very open)
        """
        # Calculate distances between finger tips
        finger_tip_landmarks = [landmarks[i] for i in self.finger_tips[1:]]  # Skip thumb
        
        total_distance = 0
        comparisons = 0
        
        for i in range(len(finger_tip_landmarks)):
            for j in range(i + 1, len(finger_tip_landmarks)):
                distance = self.calculate_distance(finger_tip_landmarks[i], finger_tip_landmarks[j])
                total_distance += distance
                comparisons += 1
        
        if comparisons > 0:
            avg_distance = total_distance / comparisons
            return min(avg_distance * 2, 1.0)  # Normalize to 0-1 range
        
        return 0.0
    
    def smooth_gesture_detection(self, gesture, confidence):
        """
        Apply temporal smoothing to reduce jitter
        
        Args:
            gesture: Current frame gesture
            confidence: Current frame confidence
            
        Returns:
            tuple: (smoothed_gesture, smoothed_confidence)
        """
        # Add current detection to history
        self.gesture_history.append((gesture, confidence))
        
        if len(self.gesture_history) < 3:
            return 'none', 0.0
        
        # Count gesture occurrences with weighted confidence
        gesture_scores = {}
        total_weight = 0
        
        for i, (g, c) in enumerate(self.gesture_history):
            # Weight recent detections more heavily
            weight = (i + 1) / len(self.gesture_history)
            score = c * weight
            
            if g not in gesture_scores:
                gesture_scores[g] = 0
            gesture_scores[g] += score
            total_weight += weight
        
        # Find best gesture
        if not gesture_scores:
            return 'none', 0.0
        
        best_gesture = max(gesture_scores.items(), key=lambda x: x[1])
        gesture_name, score = best_gesture
        
        # Normalize confidence
        normalized_confidence = score / total_weight if total_weight > 0 else 0
        
        # Only return gesture if confidence is high enough
        if normalized_confidence >= self.confidence_threshold:
            return gesture_name, normalized_confidence
        
        return 'none', 0.0    

    def handle_gesture_control(self, gesture, confidence):
        """
        Handle keyboard control based on detected gesture
        
        Args:
            gesture: Detected gesture name
            confidence: Gesture confidence
        """
        with self.key_lock:
            # Release all keys first
            self.release_all_keys_internal()
            
            if gesture == 'open_hand' and confidence >= self.confidence_threshold:
                # Press and hold ArrowUp
                self.keyboard_controller.press(Key.up)
                self.arrow_up_pressed = True
                print(f"🖐️ Open hand detected ({confidence:.2f}) → ArrowUp pressed")
            
            elif gesture == 'fist' and confidence >= self.confidence_threshold:
                # Press and hold ArrowDown
                self.keyboard_controller.press(Key.down)
                self.arrow_down_pressed = True
                print(f"✊ Fist detected ({confidence:.2f}) → ArrowDown pressed")
            
            elif gesture == 'three_fingers' and confidence >= self.confidence_threshold:
                # Press and hold ArrowLeft
                self.keyboard_controller.press(Key.left)
                self.arrow_left_pressed = True
                print(f"🤟 Three fingers detected ({confidence:.2f}) → ArrowLeft pressed")
            
            elif gesture == 'two_fingers' and confidence >= self.confidence_threshold:
                # Press and hold ArrowRight
                self.keyboard_controller.press(Key.right)
                self.arrow_right_pressed = True
                print(f"✌️ Two fingers detected ({confidence:.2f}) → ArrowRight pressed")
    
    def release_all_keys_internal(self):
        """Internal method to release all keys without lock (already locked)"""
        if self.arrow_up_pressed:
            self.keyboard_controller.release(Key.up)
            self.arrow_up_pressed = False
        
        if self.arrow_down_pressed:
            self.keyboard_controller.release(Key.down)
            self.arrow_down_pressed = False
        
        if self.arrow_left_pressed:
            self.keyboard_controller.release(Key.left)
            self.arrow_left_pressed = False
        
        if self.arrow_right_pressed:
            self.keyboard_controller.release(Key.right)
            self.arrow_right_pressed = False

    def detect_pinch(self, landmarks):
        """
        Detect pinch gesture (thumb tip close to index finger tip)
        
        Args:
            landmarks: Hand landmarks from MediaPipe
            
        Returns:
            tuple: (is_pinching, pinch_distance)
        """
        if not landmarks:
            return False, 1.0
        
        # Get thumb tip (landmark 4) and index finger tip (landmark 8)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate distance between thumb and index finger tips
        pinch_distance = self.calculate_distance(thumb_tip, index_tip)
        
        # Check if distance is below threshold
        is_pinching = pinch_distance < self.pinch_threshold
        
        return is_pinching, pinch_distance

    def get_cursor_position(self, landmarks, frame_width, frame_height):
        """
        Get cursor position based on index finger tip
        
        Args:
            landmarks: Hand landmarks from MediaPipe
            frame_width: Width of the video frame
            frame_height: Height of the video frame
            
        Returns:
            tuple: (screen_x, screen_y)
        """
        if not landmarks:
            return self.last_cursor_pos
        
        # Get index finger tip (landmark 8)
        index_tip = landmarks[8]
        
        # Convert normalized coordinates to screen coordinates
        # Flip x-coordinate for mirror effect
        screen_x = int((1 - index_tip.x) * 1920)  # Assuming 1920x1080 screen
        screen_y = int(index_tip.y * 1080)
        
        # Apply smoothing
        self.cursor_smoothing.append((screen_x, screen_y))
        
        if len(self.cursor_smoothing) > 1:
            # Average the last few positions for smoother movement
            avg_x = sum(pos[0] for pos in self.cursor_smoothing) / len(self.cursor_smoothing)
            avg_y = sum(pos[1] for pos in self.cursor_smoothing) / len(self.cursor_smoothing)
            smooth_pos = (int(avg_x), int(avg_y))
        else:
            smooth_pos = (screen_x, screen_y)
        
        self.last_cursor_pos = smooth_pos
        return smooth_pos

    def handle_cursor_control(self, landmarks, gesture, confidence, frame_width, frame_height):
        """
        Handle cursor control based on hand landmarks and gestures
        
        Args:
            landmarks: Hand landmarks from MediaPipe
            gesture: Current gesture
            confidence: Gesture confidence
            frame_width: Width of the video frame
            frame_height: Height of the video frame
        """
        if not landmarks or not self.cursor_mode:
            return
        
        # Get cursor position from index finger
        cursor_x, cursor_y = self.get_cursor_position(landmarks, frame_width, frame_height)
        
        # Move cursor
        try:
            self.mouse_controller.position = (cursor_x, cursor_y)
        except Exception as e:
            print(f"⚠️ Cursor movement error: {e}")
        
        # Detect pinch for clicking
        is_pinching, pinch_distance = self.detect_pinch(landmarks)
        
        # Handle pinch gestures (left click and drag)
        if is_pinching and not self.is_pinching:
            # Start pinch - left click
            try:
                self.mouse_controller.press(Button.left)
                self.is_pinching = True
                self.is_dragging_cursor = True
                print(f"🤏 Pinch detected ({pinch_distance:.3f}) → Left click pressed")
            except Exception as e:
                print(f"⚠️ Click error: {e}")
        
        elif not is_pinching and self.is_pinching:
            # End pinch - release left click
            try:
                self.mouse_controller.release(Button.left)
                self.is_pinching = False
                self.is_dragging_cursor = False
                print("🤏 Pinch released → Left click released")
            except Exception as e:
                print(f"⚠️ Click release error: {e}")
        
        # Handle fist gesture for right click
        if gesture == 'fist' and confidence >= self.confidence_threshold:
            try:
                self.mouse_controller.click(Button.right)
                print(f"✊ Fist detected ({confidence:.2f}) → Right click")
            except Exception as e:
                print(f"⚠️ Right click error: {e}")
        
        # Handle open hand for scrolling
        if gesture == 'open_hand' and confidence >= self.confidence_threshold:
            # Get hand center Y position for scroll detection
            hand_center_y = landmarks[9].y  # Middle finger MCP joint
            
            if self.last_hand_y != 0:
                # Calculate vertical movement
                y_movement = hand_center_y - self.last_hand_y
                
                # Accumulate scroll movement
                self.scroll_accumulator += y_movement * 10  # Scale factor
                
                # Scroll when accumulator reaches threshold
                if abs(self.scroll_accumulator) > 0.1:
                    scroll_direction = -1 if self.scroll_accumulator > 0 else 1
                    try:
                        self.mouse_controller.scroll(0, scroll_direction)
                        print(f"🖐️ Scroll {'up' if scroll_direction > 0 else 'down'}")
                        self.scroll_accumulator = 0
                    except Exception as e:
                        print(f"⚠️ Scroll error: {e}")
            
            self.last_hand_y = hand_center_y 
   
    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        self.fps_counter.append(current_time - self.last_time)
        self.last_time = current_time
        
        if len(self.fps_counter) > 1:
            avg_frame_time = sum(self.fps_counter) / len(self.fps_counter)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            return fps
        return 0
    
    def draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on frame"""
        if landmarks:
            # Draw hand landmarks
            self.mp_draw.draw_landmarks(
                frame, landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Draw finger tip circles
            h, w, _ = frame.shape
            for tip_idx in self.finger_tips:
                landmark = landmarks.landmark[tip_idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 8, (255, 255, 0), -1)
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events for interactive steering wheel
        
        Args:
            event: Mouse event type
            x, y: Mouse coordinates
            flags: Event flags
            param: Additional parameters
        """
        # Calculate distance from wheel center
        distance = math.sqrt((x - self.wheel_center_x)**2 + (y - self.wheel_center_y)**2)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is within the steering wheel
            if distance <= self.wheel_radius:
                self.is_dragging = True
                # Calculate initial angle
                self.last_mouse_angle = math.atan2(y - self.wheel_center_y, x - self.wheel_center_x)
                print("🖱️ Started dragging steering wheel")
        
        elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:
            # Update wheel rotation based on mouse movement
            if distance <= self.wheel_radius * 1.5:  # Allow some tolerance
                current_angle = math.atan2(y - self.wheel_center_y, x - self.wheel_center_x)
                angle_diff = current_angle - self.last_mouse_angle
                
                # Normalize angle difference to [-π, π]
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                
                # Update manual rotation (limit to ±90 degrees)
                self.manual_rotation += math.degrees(angle_diff)
                self.manual_rotation = max(-90, min(90, self.manual_rotation))
                
                self.last_mouse_angle = current_angle
                
                # Trigger keyboard events based on manual rotation
                self.handle_manual_steering()
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_dragging:
                self.is_dragging = False
                print("🖱️ Stopped dragging steering wheel")
                # Gradually return to center when not dragging
                self.manual_rotation *= 0.8  # Damping effect    
 
    def handle_manual_steering(self):

        """Handle keyboard events based on manual steering wheel rotation"""
        with self.key_lock:
            # Release all directional keys first
            if self.arrow_left_pressed:
                self.keyboard_controller.release(Key.left)
                self.arrow_left_pressed = False
            if self.arrow_right_pressed:
                self.keyboard_controller.release(Key.right)
                self.arrow_right_pressed = False
            
            # Apply steering based on manual rotation
            if self.manual_rotation < -15:  # Left turn threshold
                self.keyboard_controller.press(Key.left)
                self.arrow_left_pressed = True
                print(f"🖱️ Manual left turn ({self.manual_rotation:.1f}°)")
            elif self.manual_rotation > 15:  # Right turn threshold
                self.keyboard_controller.press(Key.right)
                self.arrow_right_pressed = True
                print(f"🖱️ Manual right turn ({self.manual_rotation:.1f}°)")
    
    def draw_steering_wheel(self, frame, gesture):
        """
        Draw a visual steering wheel that rotates based on gestures and manual input
        
        Args:
            frame: Video frame
            gesture: Current gesture
        """
        h, w = frame.shape[:2]
        
        # Steering wheel position (top-right corner)
        center_x = w - 120
        center_y = 120
        outer_radius = 80
        inner_radius = 15
        
        # Store wheel center for mouse interaction
        self.wheel_center_x = center_x
        self.wheel_center_y = center_y
        
        # Calculate rotation angle based on gesture or manual input
        rotation_angle = 0
        control_source = "Gesture"
        
        # Priority: Manual control > Gesture control
        if abs(self.manual_rotation) > 5:  # Manual control active
            rotation_angle = self.manual_rotation
            control_source = "Manual"
        elif gesture == 'three_fingers':  # Left turn
            rotation_angle = -45
        elif gesture == 'two_fingers':  # Right turn
            rotation_angle = 45
        
        # Convert angle to radians
        angle_rad = math.radians(rotation_angle)
        
        # Draw outer wheel rim with different color based on control source
        if control_source == "Manual":
            wheel_color = (0, 150, 255)  # Orange for manual control
        elif self.is_dragging:
            wheel_color = (0, 255, 255)  # Yellow when dragging
        else:
            wheel_color = (100, 100, 100)  # Gray for gesture control
        
        cv2.circle(frame, (center_x, center_y), outer_radius, wheel_color, 8)
        
        # Draw interactive area indicator
        if self.is_dragging:
            cv2.circle(frame, (center_x, center_y), outer_radius + 5, (0, 255, 255), 2)
        
        # Draw inner hub
        hub_color = (50, 50, 50)  # Dark gray
        cv2.circle(frame, (center_x, center_y), inner_radius, hub_color, -1)    
    
        # Draw spokes (rotated based on gesture or manual input)
        spoke_color = (150, 150, 150)  # Light gray
        spoke_length = outer_radius - 10
        
        # Calculate spoke endpoints with rotation
        for i in range(4):  # 4 spokes
            base_angle = i * math.pi / 2  # 90 degrees apart
            final_angle = base_angle + angle_rad
            
            # Calculate spoke endpoints
            end_x = center_x + int(spoke_length * math.cos(final_angle))
            end_y = center_y + int(spoke_length * math.sin(final_angle))
            
            cv2.line(frame, (center_x, center_y), (end_x, end_y), spoke_color, 4)
        
        # Draw grip areas on the wheel (also rotated)
        grip_color = (80, 80, 80)  # Darker gray
        grip_radius = outer_radius - 5
        
        # Top and bottom grips
        for grip_angle in [math.pi/2, -math.pi/2]:  # Top and bottom
            final_grip_angle = grip_angle + angle_rad
            grip_x = center_x + int(grip_radius * math.cos(final_grip_angle))
            grip_y = center_y + int(grip_radius * math.sin(final_grip_angle))
            cv2.circle(frame, (grip_x, grip_y), 12, grip_color, -1)
        
        # Draw directional indicators around the wheel
        indicator_radius = outer_radius + 20
        
        # Up arrow (accelerate)
        up_color = (0, 255, 0) if gesture == 'open_hand' else (100, 100, 100)
        up_x = center_x
        up_y = center_y - indicator_radius
        cv2.arrowedLine(frame, (up_x, up_y + 10), (up_x, up_y - 10), up_color, 3)
        
        # Down arrow (brake)
        down_color = (0, 0, 255) if gesture == 'fist' else (100, 100, 100)
        down_x = center_x
        down_y = center_y + indicator_radius
        cv2.arrowedLine(frame, (down_x, down_y - 10), (down_x, down_y + 10), down_color, 3)
        
        # Left arrow (enhanced for manual control)
        left_active = gesture == 'three_fingers' or (control_source == "Manual" and rotation_angle < -15)
        left_color = (255, 255, 0) if left_active else (100, 100, 100)
        left_x = center_x - indicator_radius
        left_y = center_y
        cv2.arrowedLine(frame, (left_x + 10, left_y), (left_x - 10, left_y), left_color, 3)
        
        # Right arrow (enhanced for manual control)
        right_active = gesture == 'two_fingers' or (control_source == "Manual" and rotation_angle > 15)
        right_color = (255, 0, 255) if right_active else (100, 100, 100)
        right_x = center_x + indicator_radius
        right_y = center_y
        cv2.arrowedLine(frame, (right_x - 10, right_y), (right_x + 10, right_y), right_color, 3)
        
        # Draw steering wheel label with control source
        cv2.putText(frame, f'STEERING ({control_source})', (center_x - 60, center_y + outer_radius + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw rotation angle
        cv2.putText(frame, f'{rotation_angle:.1f}°', (center_x - 20, center_y + outer_radius + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw interaction hint
        if not self.is_dragging:
            cv2.putText(frame, 'Click & Drag', (center_x - 35, center_y + outer_radius + 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)    
    
    def draw_ui(self, frame, gesture, confidence, fps, extended_count=0):
        """
        Draw UI elements on frame
        
        Args:
            frame: Video frame
            gesture: Current gesture
            confidence: Gesture confidence
            fps: Current FPS
            extended_count: Number of extended fingers
        """
        h, w = frame.shape[:2]
        
        # Draw FPS
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw current gesture
        gesture_color = (0, 255, 255) if gesture != 'none' else (128, 128, 128)
        cv2.putText(frame, f'Gesture: {gesture}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
        
        # Draw confidence
        cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
        
        # Draw extended fingers count
        cv2.putText(frame, f'Extended Fingers: {extended_count}', (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw control mode
        mode_color = (0, 255, 255) if self.cursor_mode else (255, 255, 0)
        mode_text = "CURSOR MODE" if self.cursor_mode else "GAME MODE"
        cv2.putText(frame, f'Mode: {mode_text}', (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        if self.cursor_mode:
            # Draw cursor control states
            pinch_color = (0, 255, 0) if self.is_pinching else (128, 128, 128)
            drag_color = (0, 255, 0) if self.is_dragging_cursor else (128, 128, 128)
            
            cv2.putText(frame, f'Pinch: {"ON" if self.is_pinching else "OFF"}', 
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pinch_color, 2)
            cv2.putText(frame, f'Drag: {"ON" if self.is_dragging_cursor else "OFF"}', 
                       (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, drag_color, 2)
            cv2.putText(frame, f'Cursor: ({self.last_cursor_pos[0]}, {self.last_cursor_pos[1]})', 
                       (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # Draw key states for game mode
            up_color = (0, 255, 0) if self.arrow_up_pressed else (128, 128, 128)
            down_color = (0, 255, 0) if self.arrow_down_pressed else (128, 128, 128)
            left_color = (0, 255, 0) if self.arrow_left_pressed else (128, 128, 128)
            right_color = (0, 255, 0) if self.arrow_right_pressed else (128, 128, 128)
            
            cv2.putText(frame, f'ArrowUp: {"ON" if self.arrow_up_pressed else "OFF"}', 
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, up_color, 2)
            cv2.putText(frame, f'ArrowDown: {"ON" if self.arrow_down_pressed else "OFF"}', 
                       (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, down_color, 2)
            cv2.putText(frame, f'ArrowLeft: {"ON" if self.arrow_left_pressed else "OFF"}', 
                       (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 2)
            cv2.putText(frame, f'ArrowRight: {"ON" if self.arrow_right_pressed else "OFF"}', 
                       (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 2)
        
        # Draw steering wheel
        self.draw_steering_wheel(frame, gesture)
        
        # Draw instructions based on current mode
        if self.cursor_mode:
            instructions = [
                "🖱️ CURSOR MODE (Press 'c' to toggle):",
                "👆 Index finger → Move cursor",
                "🤏 Pinch (thumb + index) → Left click & drag",
                "✊ Fist → Right click",
                "🖐️ Open hand up/down → Scroll",
                "Press 'q' to quit"
            ]
        else:
            instructions = [
                "🎮 GAME MODE (Press 'c' to toggle):",
                "🤟 Three Fingers → ArrowLeft",
                "✌️ Two Fingers → ArrowRight",
                "🖐️ Open Hand → ArrowUp (Boost)",
                "✊ Fist → ArrowDown (Brake)",
                "Press 'q' to quit"
            ]
        
        for i, instruction in enumerate(instructions):
            y_pos = h - 20 - (len(instructions) - i - 1) * 25
            cv2.putText(frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw gesture status box
        status_color = (0, 255, 0) if gesture in ['open_hand', 'fist', 'three_fingers', 'two_fingers'] else (0, 0, 255)
        cv2.rectangle(frame, (w - 250, 250), (w - 10, 350), status_color, 2)
        
        status_text = "ACTIVE" if gesture in ['open_hand', 'fist', 'three_fingers', 'two_fingers'] else "WAITING"
        cv2.putText(frame, status_text, (w - 230, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2) 
   
    def release_all_keys(self):
        """Release all pressed keys"""
        with self.key_lock:
            if self.arrow_up_pressed:
                self.keyboard_controller.release(Key.up)
                self.arrow_up_pressed = False
            
            if self.arrow_down_pressed:
                self.keyboard_controller.release(Key.down)
                self.arrow_down_pressed = False
            
            if self.arrow_left_pressed:
                self.keyboard_controller.release(Key.left)
                self.arrow_left_pressed = False
            
            if self.arrow_right_pressed:
                self.keyboard_controller.release(Key.right)
                self.arrow_right_pressed = False
    
    def run(self, camera_index=0):
        """
        Main detection and control loop
        
        Args:
            camera_index: Camera device index
        """
        print("🚀 Starting MediaPipe Gesture Control System...")
        print("📹 Initializing camera...")
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties for optimal performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        print("✅ Camera initialized successfully")
        print("🎮 Focus your game window and start using gestures!")
        print("📋 Show your hand clearly to the camera")
        print("Press 'q' in the video window to quit")
        
        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with MediaPipe
                results = self.hands.process(rgb_frame)
                
                # Initialize variables
                gesture = 'none'
                confidence = 0.0
                extended_count = 0
                
                # Process hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        self.draw_landmarks(frame, hand_landmarks)
                        
                        # Classify gesture
                        gesture, confidence = self.classify_gesture(hand_landmarks.landmark)
                        
                        # Count extended fingers for display
                        extended_count, _ = self.count_extended_fingers(hand_landmarks.landmark)
                        
                        break  # Only process first hand      
          
                # Apply smoothing
                smooth_gesture, smooth_confidence = self.smooth_gesture_detection(gesture, confidence)
                
                # Update current state
                self.current_gesture = smooth_gesture
                self.gesture_confidence = smooth_confidence
                
                # Handle control based on current mode
                if self.cursor_mode:
                    # Handle cursor control
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.handle_cursor_control(
                                hand_landmarks.landmark, 
                                smooth_gesture, 
                                smooth_confidence,
                                frame.shape[1],  # width
                                frame.shape[0]   # height
                            )
                            break
                else:
                    # Handle keyboard control for game mode
                    self.handle_gesture_control(smooth_gesture, smooth_confidence)
                
                # Calculate FPS
                fps = self.calculate_fps()
                
                # Draw UI
                self.draw_ui(frame, smooth_gesture, smooth_confidence, fps, extended_count)
                
                # Display frame
                cv2.imshow('MediaPipe Gesture Control', frame)
                
                # Set mouse callback for interactive steering wheel
                cv2.setMouseCallback('MediaPipe Gesture Control', self.mouse_callback)
                
                # Apply damping to manual rotation when not dragging
                if not self.is_dragging and abs(self.manual_rotation) > 1:
                    self.manual_rotation *= 0.95  # Gradual return to center
                    if abs(self.manual_rotation) < 1:
                        self.manual_rotation = 0
                
                # Check for quit and mode toggle
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):  # Toggle cursor mode
                    self.cursor_mode = not self.cursor_mode
                    mode_text = "CURSOR MODE" if self.cursor_mode else "GAME MODE"
                    print(f"🔄 Switched to {mode_text}")
                    # Release all keys and mouse buttons when switching modes
                    self.release_all_keys()
                    if self.is_pinching:
                        try:
                            self.mouse_controller.release(Button.left)
                            self.is_pinching = False
                            self.is_dragging_cursor = False
                        except:
                            pass
                elif key == ord('r'):  # Reset manual rotation
                    self.manual_rotation = 0
                    print("🔄 Manual steering reset to center")
                    
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user")
        except Exception as e:
            print(f"❌ Error during detection: {e}")
        finally:
            # Cleanup
            print("🧹 Cleaning up...")
            self.release_all_keys()
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Cleanup completed")

def main():
    parser = argparse.ArgumentParser(description='MediaPipe Hand Gesture Control System with Interactive Steering')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Gesture classification confidence threshold (default: 0.7)')
    parser.add_argument('--detection-confidence', type=float, default=0.7,
                       help='MediaPipe hand detection confidence (default: 0.7)')
    parser.add_argument('--tracking-confidence', type=float, default=0.5,
                       help='MediaPipe hand tracking confidence (default: 0.5)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    
    args = parser.parse_args()
    
    print("🖐️ MediaPipe Hand Gesture Control System with Interactive Steering")
    print(f"⚙️ Confidence threshold: {args.confidence}")
    print(f"🔍 Detection confidence: {args.detection_confidence}")
    print(f"📍 Tracking confidence: {args.tracking_confidence}")
    print(f"📷 Camera index: {args.camera}")
    print()
    
    # Create and run controller
    controller = MediaPipeGestureController(
        confidence_threshold=args.confidence,
        detection_confidence=args.detection_confidence,
        tracking_confidence=args.tracking_confidence
    )
    
    controller.run(camera_index=args.camera)

if __name__ == "__main__":
    main()