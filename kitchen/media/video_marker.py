
import os
import csv
import traceback
import cv2
from tqdm import tqdm

from kitchen.configs import routing
from kitchen.settings.fluorescence import DEFAULT_RECORDING_DURATION


def marker_video_use_timeline(dir_path: str):
    """Mark videos with event text overlays based on timeline data."""
    video_files = routing.search_pattern_file('VIDEO_*', dir_path)
    timeline_files = routing.search_pattern_file('TIMELINE_*', dir_path)
    demo_files = routing.search_pattern_file('DEMO_*', dir_path)
    
    # Match files by basename (without extensions)
    existing_demos = {os.path.splitext(os.path.basename(f)[5:])[0] for f in demo_files}  # Remove "DEMO_" prefix and extension
    
    for video_file in video_files:
        basename = os.path.splitext(os.path.basename(video_file)[6:])[0]  # Remove "VIDEO_" prefix and extension
        
        if basename in existing_demos:
            continue  # Skip if demo already exists
            
        # Find matching timeline file
        timeline_file = None
        for tf in timeline_files:
            if os.path.splitext(os.path.basename(tf)[9:])[0] == basename:  # Remove "TIMELINE_" prefix and extension
                timeline_file = tf
                break
        
        if not timeline_file:
            print(f"No timeline found for {basename}")
            continue
            
        # Parse timeline and create demo
        try:
            events = parse_timeline(timeline_file)
            demo_path = os.path.join(os.path.dirname(video_file), f"DEMO_{basename}.avi")
            create_demo_video(video_file, demo_path, events)
            print(f"Created: {demo_path}")
        except Exception as e:
            print(f"Error processing {basename}: {e}")
            traceback.print_exc()


def parse_timeline(timeline_file: str):
    """Parse timeline file and return event periods."""
    events = {}
    all_events = []
    
    with open(timeline_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_val = float(row['time'])
            detail = row['details'].strip()
            
            if detail == 'task start':
                events['start'] = time_val
            elif detail == 'task end':
                events['end'] = time_val
            else:
                all_events.append((time_val, detail))
                
    if "start" not in events:
        events['start'] = 0
    if "end" not in events:
        events['end'] = events['start'] + DEFAULT_RECORDING_DURATION

    # Sort events by time
    all_events.sort(key=lambda x: x[0])
    
    # Create event periods by matching On/Off pairs
    periods = []
    active_events = {}  # Track currently active events
    
    for time_val, detail in all_events:
        if detail.endswith('On'):
            event_name = detail[:-2]
            active_events[event_name] = time_val
        elif detail.endswith('Off'):
            event_name = detail[:-3]
            if event_name in active_events:
                start_time = active_events.pop(event_name)
                periods.append((event_name, start_time, time_val))
    
    events['periods'] = periods
    return events


def create_demo_video(video_file: str, demo_file: str, events: dict):
    """Create demo video with enhanced text overlays."""
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(demo_file, fourcc, fps, (width, height))
    
    start_time = events['start']
    end_time = events['end']
    video_duration = end_time - start_time
    min_duration_frames = int(0.5 * fps)  # Minimum 0.5s display
    font_scale, thickness = 1.8, 2
    
    # Event colors (BGR format for OpenCV)
    event_colors = {
        'Buzzer': (255, 0, 255),      # Magenta
        'Water': (255, 255, 0),       # Cyan  
        'NoWater': (128, 128, 128),   # Gray
        'VerticalPuff': (0, 255, 0),  # Green
        'HorizontalPuff': (0, 128, 255), # Orange
        'PeltierLeft': (255, 128, 0), # Blue
        'PeltierRight': (255, 128, 0), # Blue
        'PeltierBoth': (255, 128, 0), # Blue
        'FakeRelay': (0, 0, 255),     # Red
    }
    
    # Extend short events to minimum duration
    extended_periods = []
    for event_name, event_start, event_end in events['periods']:
        duration_frames = int((event_end - event_start) / video_duration * total_frames)
        if duration_frames < min_duration_frames:
            # Extend the event to minimum duration
            extended_end_time = event_start + (min_duration_frames / total_frames) * video_duration
            extended_periods.append((event_name, event_start, extended_end_time))
        else:
            extended_periods.append((event_name, event_start, event_end))
    
    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_file)}") as pbar:
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Calculate current timeline time
            progress = frame_num / total_frames
            current_time = start_time + progress * video_duration
            
            # Create overlay for text with transparency
            overlay = frame.copy()
            
            # Add text for active events
            y_pos = 50
            for event_name, event_start, event_end in extended_periods:
                if event_start <= current_time <= event_end:
                    color = event_colors.get(event_name, (0, 255, 0))  # Default green
                    
                    # Get text dimensions
                    (text_width, text_height), baseline = cv2.getTextSize(event_name, cv2.FONT_HERSHEY_COMPLEX, font_scale, thickness)
                    
                    # Add background rectangle for better readability
                    padding = 10
                    cv2.rectangle(overlay, 
                                 (10, y_pos - text_height - padding), 
                                 (10 + text_width + 2*padding, y_pos + baseline + padding), 
                                 (0, 0, 0), -1)
                    
                    # Add text
                    cv2.putText(overlay, event_name, (10 + padding, y_pos), 
                               cv2.FONT_HERSHEY_COMPLEX, font_scale, color, thickness, cv2.LINE_AA)
                    y_pos += text_height + 2*padding + 10
            
            # Blend overlay with original frame for transparency
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            out.write(frame)
            frame_num += 1
            pbar.update(1)
    
    cap.release()
    out.release()
