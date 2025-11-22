#!/usr/bin/env python3
"""
Pipeline script that chains sber-swap face swap and mediapipe sticker overlay.

This script:
1. Takes an input video and runs face swap using sber-swap
2. Takes the face-swapped video and adds stickers using mediapipe
3. Produces a final output video with both face swap and stickers
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sber-swap'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mediapipe'))


def get_results_paths(output_path: str, use_results_dir: bool = True):
    """
    Get paths for intermediate and final output videos.
    
    Args:
        output_path: Desired final output path
        use_results_dir: If True, organize outputs in results/ directory
    
    Returns:
        tuple: (intermediate_path, final_path)
    """
    if use_results_dir:
        # Get base directory of pipeline script (project root)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, 'results')
        intermediate_dir = os.path.join(results_dir, 'intermediate')
        final_dir = os.path.join(results_dir, 'final')
        
        # Create directories if they don't exist
        os.makedirs(intermediate_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)
        
        # Generate filenames with timestamp
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        intermediate_path = os.path.join(
            intermediate_dir,
            f"{base_name}_intermediate_{timestamp}.mp4"
        )
        final_path = os.path.join(
            final_dir,
            f"{base_name}_final_{timestamp}.mp4"
        )
    else:
        # Use same directory as output, with intermediate prefix
        output_dir = os.path.dirname(output_path) or '.'
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        intermediate_path = os.path.join(
            output_dir,
            f".pipeline_intermediate_{base_name}.mp4"
        )
        final_path = output_path
    
    return intermediate_path, final_path


def run_sber_swap(input_video: str, 
                  source_paths: list,
                  target_faces_paths: list,
                  output_video: str,
                  sber_swap_dir: str = None,
                  **kwargs) -> str:
    """
    Run sber-swap face swap on input video
    
    Args:
        input_video: Path to input video
        source_paths: List of source face image paths
        target_faces_paths: List of target face image paths (can be empty)
        output_video: Path to output face-swapped video
        sber_swap_dir: Directory containing sber-swap (default: ./sber-swap)
        **kwargs: Additional arguments for sber-swap inference
    
    Returns:
        Path to the face-swapped video
    """
    if sber_swap_dir is None:
        sber_swap_dir = os.path.join(os.path.dirname(__file__), 'sber-swap')
    
    inference_script = os.path.join(sber_swap_dir, 'inference.py')
    
    if not os.path.exists(inference_script):
        raise FileNotFoundError(f"sber-swap inference script not found: {inference_script}")
    
    # Convert paths to absolute to avoid issues when changing directories
    input_video = os.path.abspath(input_video)
    output_video = os.path.abspath(output_video)
    source_paths = [os.path.abspath(p) for p in source_paths]
    target_faces_paths = [os.path.abspath(p) for p in target_faces_paths]
    
    # Build command
    cmd = [
        sys.executable,
        inference_script,
        '--target_video', input_video,
        '--out_video_name', output_video,
    ]
    
    # Add source paths
    if source_paths:
        cmd.extend(['--source_paths'] + source_paths)
    
    # Add target face paths (optional)
    if target_faces_paths:
        cmd.extend(['--target_faces_paths'] + target_faces_paths)
    
    # Add optional arguments
    if 'G_path' in kwargs:
        cmd.extend(['--G_path', kwargs['G_path']])
    if 'batch_size' in kwargs:
        cmd.extend(['--batch_size', str(kwargs['batch_size'])])
    if 'use_sr' in kwargs and kwargs['use_sr']:
        cmd.extend(['--use_sr', 'True'])
    if 'similarity_th' in kwargs:
        cmd.extend(['--similarity_th', str(kwargs['similarity_th'])])
    
    print("=" * 60)
    print("Running sber-swap face swap...")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Change to sber-swap directory to ensure relative paths work
    original_dir = os.getcwd()
    try:
        os.chdir(sber_swap_dir)
        subprocess.run(cmd, check=True, capture_output=False)
    finally:
        os.chdir(original_dir)
    
    if not os.path.exists(output_video):
        raise RuntimeError(f"sber-swap failed to create output video: {output_video}")
    
    print(f"\n✓ Face swap completed: {output_video}\n")
    return output_video


def run_mediapipe_stickers(input_video: str,
                          output_video: str,
                          sticker_configs: list,
                          mediapipe_dir: str = None,
                          **kwargs) -> str:
    """
    Run mediapipe sticker overlay on input video
    
    Args:
        input_video: Path to input video (face-swapped video)
        output_video: Path to final output video with stickers
        sticker_configs: List of sticker config strings (format: "path:position[:scale[:rotation]]")
        mediapipe_dir: Directory containing mediapipe (default: ./mediapipe)
        **kwargs: Additional arguments for mediapipe
    
    Returns:
        Path to the final video with stickers
    """
    if mediapipe_dir is None:
        mediapipe_dir = os.path.join(os.path.dirname(__file__), 'mediapipe')
    
    sticker_script = os.path.join(mediapipe_dir, 'add_stickers_to_video.py')
    
    if not os.path.exists(sticker_script):
        raise FileNotFoundError(f"mediapipe sticker script not found: {sticker_script}")
    
    # Convert paths to absolute to avoid issues when changing directories
    input_video = os.path.abspath(input_video)
    output_video = os.path.abspath(output_video)
    
    # Process sticker configs to convert relative paths to absolute
    processed_sticker_configs = []
    for sticker_config in sticker_configs:
        parts = sticker_config.split(':')
        if len(parts) >= 1:
            # Convert sticker path to absolute
            sticker_path = parts[0]
            if not os.path.isabs(sticker_path):
                sticker_path = os.path.abspath(sticker_path)
            # Reconstruct config with absolute path
            parts[0] = sticker_path
            processed_sticker_configs.append(':'.join(parts))
        else:
            processed_sticker_configs.append(sticker_config)
    
    # Build command
    cmd = [
        sys.executable,
        sticker_script,
        '-i', input_video,
        '-o', output_video,
    ]
    
    # Add sticker configurations
    for sticker_config in processed_sticker_configs:
        cmd.extend(['-s', sticker_config])
    
    # Add optional arguments
    if 'min_detection_confidence' in kwargs:
        cmd.extend(['--min-detection-confidence', str(kwargs['min_detection_confidence'])])
    if 'min_tracking_confidence' in kwargs:
        cmd.extend(['--min-tracking-confidence', str(kwargs['min_tracking_confidence'])])
    if 'codec' in kwargs:
        cmd.extend(['--codec', kwargs['codec']])
    
    print("=" * 60)
    print("Running mediapipe sticker overlay...")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Change to mediapipe directory
    original_dir = os.getcwd()
    try:
        os.chdir(mediapipe_dir)
        subprocess.run(cmd, check=True, capture_output=False)
    finally:
        os.chdir(original_dir)
    
    if not os.path.exists(output_video):
        raise RuntimeError(f"mediapipe failed to create output video: {output_video}")
    
    print(f"\n✓ Sticker overlay completed: {output_video}\n")
    return output_video


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline: Face swap (sber-swap) → Sticker overlay (mediapipe)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with one source face and one sticker
  python pipeline.py -i input.mp4 -o output.mp4 \\
    --source_paths source_face.jpg \\
    --stickers "sticker.png:forehead:1.5"
  
  # Multiple source faces and multiple stickers
  python pipeline.py -i input.mp4 -o output.mp4 \\
    --source_paths face1.jpg face2.jpg \\
    --target_faces_paths target1.jpg target2.jpg \\
    --stickers "sticker1.png:forehead:1.5" "sticker2.png:nose:1.0"
  
  # Keep intermediate face-swapped video
  python pipeline.py -i input.mp4 -o output.mp4 \\
    --source_paths source_face.jpg \\
    --stickers "sticker.png:forehead:1.5" \\
    --keep_intermediate

Valid sticker positions: forehead, nose, left_cheek, right_cheek, left_eye, right_eye, chin, mouth
        """
    )
    
    # Required arguments
    parser.add_argument('-i', '--input', required=True,
                       help='Input video path')
    parser.add_argument('-o', '--output', required=True,
                       help='Final output video path (with face swap and stickers)')
    parser.add_argument('--source_paths', required=True, nargs='+',
                       help='Path(s) to source face image(s) for face swap')
    
    # Optional sber-swap arguments
    parser.add_argument('--target_faces_paths', nargs='+', default=[],
                       help='Path(s) to target face image(s) in video (optional)')
    parser.add_argument('--G_path', 
                       default='weights/G_unet_2blocks.pth',
                       help='Path to sber-swap generator weights (relative to sber-swap dir)')
    parser.add_argument('--batch_size', type=int, default=40,
                       help='Batch size for sber-swap inference')
    parser.add_argument('--use_sr', action='store_true',
                       help='Enable super resolution in sber-swap')
    parser.add_argument('--similarity_th', type=float, default=0.15,
                       help='Similarity threshold for sber-swap face matching')
    
    # Required mediapipe arguments
    parser.add_argument('-s', '--stickers', required=True, action='append',
                       help='Sticker configuration: path:position[:scale[:rotation]] (can specify multiple times)')
    
    # Optional mediapipe arguments
    parser.add_argument('--min-detection-confidence', type=float, default=0.5,
                       help='Minimum confidence for face detection (0.0-1.0)')
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5,
                       help='Minimum confidence for face tracking (0.0-1.0)')
    parser.add_argument('--codec', type=str, default='mp4v',
                       help='Video codec (default: mp4v)')
    
    # Pipeline options
    parser.add_argument('--keep_intermediate', action='store_true',
                       help='Keep the intermediate face-swapped video (before stickers)')
    parser.add_argument('--intermediate_output', type=str, default=None,
                       help='Path for intermediate face-swapped video (default: auto-generated)')
    parser.add_argument('--use_results_dir', action='store_true',
                       help='Organize outputs in test/results/ directory structure')
    parser.add_argument('--sber_swap_dir', type=str, default=None,
                       help='Path to sber-swap directory (default: ./sber-swap)')
    parser.add_argument('--mediapipe_dir', type=str, default=None,
                       help='Path to mediapipe directory (default: ./mediapipe)')
    
    args = parser.parse_args()
    
    # Validate input video exists
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        return 1
    
    # Validate source images exist
    for source_path in args.source_paths:
        if not os.path.exists(source_path):
            print(f"Error: Source image not found: {source_path}")
            return 1
    
    # Validate target images if provided
    for target_path in args.target_faces_paths:
        if not os.path.exists(target_path):
            print(f"Error: Target face image not found: {target_path}")
            return 1
    
    # Create intermediate video path if not specified
    if args.intermediate_output:
        intermediate_video = args.intermediate_output
    elif args.use_results_dir:
        # Use results directory structure
        intermediate_video, final_output = get_results_paths(
            args.output, use_results_dir=True
        )
        # Update final output path to use results directory
        args.output = final_output
    else:
        # Create temp file in same directory as output
        output_dir = os.path.dirname(args.output) or '.'
        intermediate_video = os.path.join(
            output_dir,
            f".pipeline_intermediate_{os.path.basename(args.output)}"
        )
    
    try:
        # Step 1: Run sber-swap face swap
        print("\n" + "=" * 60)
        print("STEP 1: Face Swap (sber-swap)")
        print("=" * 60 + "\n")
        
        sber_swap_kwargs = {
            'G_path': args.G_path,
            'batch_size': args.batch_size,
            'use_sr': args.use_sr,
            'similarity_th': args.similarity_th,
        }
        
        face_swapped_video = run_sber_swap(
            input_video=args.input,
            source_paths=args.source_paths,
            target_faces_paths=args.target_faces_paths,
            output_video=intermediate_video,
            sber_swap_dir=args.sber_swap_dir,
            **sber_swap_kwargs
        )
        
        # Step 2: Run mediapipe sticker overlay
        print("\n" + "=" * 60)
        print("STEP 2: Sticker Overlay (mediapipe)")
        print("=" * 60 + "\n")
        
        mediapipe_kwargs = {
            'min_detection_confidence': args.min_detection_confidence,
            'min_tracking_confidence': args.min_tracking_confidence,
            'codec': args.codec,
        }
        
        final_video = run_mediapipe_stickers(
            input_video=face_swapped_video,
            output_video=args.output,
            sticker_configs=args.stickers,
            mediapipe_dir=args.mediapipe_dir,
            **mediapipe_kwargs
        )
        
        # Clean up intermediate file if not keeping it
        if not args.keep_intermediate and os.path.exists(intermediate_video):
            print(f"Cleaning up intermediate file: {intermediate_video}")
            os.remove(intermediate_video)
        
        print("=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
        print(f"Final output: {final_video}")
        if args.keep_intermediate:
            print(f"Intermediate (face-swapped): {face_swapped_video}")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\nError: Pipeline step failed with return code {e.returncode}")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

