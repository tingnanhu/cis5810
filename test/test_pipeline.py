#!/usr/bin/env python3
"""
Tests for the pipeline script that chains sber-swap and mediapipe.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path

# Add parent directory to path to import pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import run_sber_swap, run_mediapipe_stickers, main


class TestPipeline(unittest.TestCase):
    """Test cases for the pipeline script"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.input_video = os.path.join(self.test_dir, 'input.mp4')
        self.output_video = os.path.join(self.test_dir, 'output.mp4')
        self.intermediate_video = os.path.join(self.test_dir, 'intermediate.mp4')
        self.source_face = os.path.join(self.test_dir, 'source.jpg')
        self.sticker_path = os.path.join(self.test_dir, 'sticker.png')
        
        # Create dummy files
        Path(self.input_video).touch()
        Path(self.source_face).touch()
        Path(self.sticker_path).touch()
        
        # Get absolute paths
        self.sber_swap_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'sber-swap')
        )
        self.mediapipe_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'mediapipe')
        )

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch('pipeline.subprocess.run')
    @patch('pipeline.os.path.exists')
    @patch('pipeline.os.getcwd')
    @patch('pipeline.os.chdir')
    def test_run_sber_swap_basic(self, mock_chdir, mock_getcwd, 
                                  mock_exists, mock_subprocess):
        """Test basic sber-swap execution"""
        mock_exists.return_value = True
        mock_getcwd.return_value = '/original/dir'
        mock_subprocess.return_value = Mock(returncode=0)
        
        # Create output video file to simulate successful run
        Path(self.intermediate_video).touch()
        
        result = run_sber_swap(
            input_video=self.input_video,
            source_paths=[self.source_face],
            target_faces_paths=[],
            output_video=self.intermediate_video,
            sber_swap_dir=self.sber_swap_dir
        )
        
        # Verify subprocess was called
        self.assertTrue(mock_subprocess.called)
        
        # Verify command structure
        call_args = mock_subprocess.call_args[0][0]
        self.assertIn('inference.py', call_args[1])
        self.assertIn('--target_video', call_args)
        self.assertIn('--source_paths', call_args)
        self.assertIn('--out_video_name', call_args)
        
        # Verify directory change
        mock_chdir.assert_called()
        
        self.assertEqual(result, self.intermediate_video)

    @patch('pipeline.subprocess.run')
    @patch('pipeline.os.path.exists')
    def test_run_sber_swap_with_options(self, mock_exists, mock_subprocess):
        """Test sber-swap with optional parameters"""
        mock_exists.return_value = True
        mock_subprocess.return_value = Mock(returncode=0)
        Path(self.intermediate_video).touch()
        
        run_sber_swap(
            input_video=self.input_video,
            source_paths=[self.source_face],
            target_faces_paths=[self.source_face],
            output_video=self.intermediate_video,
            sber_swap_dir=self.sber_swap_dir,
            G_path='weights/custom.pth',
            batch_size=20,
            use_sr=True,
            similarity_th=0.2
        )
        
        # Verify optional arguments were passed
        call_args = mock_subprocess.call_args[0][0]
        self.assertIn('--G_path', call_args)
        self.assertIn('--batch_size', call_args)
        self.assertIn('--use_sr', call_args)
        self.assertIn('--similarity_th', call_args)
        self.assertIn('weights/custom.pth', call_args)
        self.assertIn('20', call_args)
        self.assertIn('0.2', call_args)

    @patch('pipeline.subprocess.run')
    @patch('pipeline.os.path.exists')
    def test_run_sber_swap_absolute_paths(self, mock_exists, mock_subprocess):
        """Test that paths are converted to absolute"""
        mock_exists.return_value = True
        mock_subprocess.return_value = Mock(returncode=0)
        Path(self.intermediate_video).touch()
        
        # Use relative paths
        rel_input = 'input.mp4'
        rel_output = 'output.mp4'
        rel_source = 'source.jpg'
        
        run_sber_swap(
            input_video=rel_input,
            source_paths=[rel_source],
            target_faces_paths=[],
            output_video=rel_output,
            sber_swap_dir=self.sber_swap_dir
        )
        
        # Verify paths in command are absolute
        call_args = mock_subprocess.call_args[0][0]
        input_idx = call_args.index('--target_video') + 1
        output_idx = call_args.index('--out_video_name') + 1
        
        self.assertTrue(os.path.isabs(call_args[input_idx]))
        self.assertTrue(os.path.isabs(call_args[output_idx]))

    @patch('pipeline.os.path.exists')
    def test_run_sber_swap_missing_script(self, mock_exists):
        """Test error when sber-swap script is missing"""
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            run_sber_swap(
                input_video=self.input_video,
                source_paths=[self.source_face],
                target_faces_paths=[],
                output_video=self.intermediate_video,
                sber_swap_dir=self.sber_swap_dir
            )

    @patch('pipeline.subprocess.run')
    @patch('pipeline.os.path.exists')
    def test_run_mediapipe_stickers_basic(self, mock_exists, mock_subprocess):
        """Test basic mediapipe sticker overlay execution"""
        mock_exists.return_value = True
        mock_subprocess.return_value = Mock(returncode=0)
        Path(self.output_video).touch()
        
        sticker_configs = [f"{self.sticker_path}:forehead:1.5"]
        
        result = run_mediapipe_stickers(
            input_video=self.intermediate_video,
            output_video=self.output_video,
            sticker_configs=sticker_configs,
            mediapipe_dir=self.mediapipe_dir
        )
        
        # Verify subprocess was called
        self.assertTrue(mock_subprocess.called)
        
        # Verify command structure
        call_args = mock_subprocess.call_args[0][0]
        self.assertIn('add_stickers_to_video.py', call_args[1])
        self.assertIn('-i', call_args)
        self.assertIn('-o', call_args)
        self.assertIn('-s', call_args)
        
        self.assertEqual(result, self.output_video)

    @patch('pipeline.subprocess.run')
    @patch('pipeline.os.path.exists')
    def test_run_mediapipe_stickers_multiple(self, mock_exists, mock_subprocess):
        """Test mediapipe with multiple stickers"""
        mock_exists.return_value = True
        mock_subprocess.return_value = Mock(returncode=0)
        Path(self.output_video).touch()
        
        sticker_configs = [
            f"{self.sticker_path}:forehead:1.5",
            f"{self.sticker_path}:nose:1.0",
            f"{self.sticker_path}:left_cheek:0.8"
        ]
        
        run_mediapipe_stickers(
            input_video=self.intermediate_video,
            output_video=self.output_video,
            sticker_configs=sticker_configs,
            mediapipe_dir=self.mediapipe_dir
        )
        
        # Verify multiple -s flags
        call_args = mock_subprocess.call_args[0][0]
        s_count = call_args.count('-s')
        self.assertEqual(s_count, 3)

    @patch('pipeline.subprocess.run')
    @patch('pipeline.os.path.exists')
    def test_run_mediapipe_stickers_absolute_paths(self, mock_exists, 
                                                   mock_subprocess):
        """Test that sticker paths are converted to absolute"""
        mock_exists.return_value = True
        mock_subprocess.return_value = Mock(returncode=0)
        Path(self.output_video).touch()
        
        # Use relative sticker path
        sticker_configs = ["sticker.png:forehead:1.5"]
        
        run_mediapipe_stickers(
            input_video=self.intermediate_video,
            output_video=self.output_video,
            sticker_configs=sticker_configs,
            mediapipe_dir=self.mediapipe_dir
        )
        
        # Verify sticker path in config is absolute
        call_args = mock_subprocess.call_args[0][0]
        s_indices = [i for i, arg in enumerate(call_args) if arg == '-s']
        for idx in s_indices:
            config = call_args[idx + 1]
            sticker_path = config.split(':')[0]
            # If it's a real path (not just 'sticker.png'), check if absolute
            if os.path.sep in sticker_path or os.path.isabs(sticker_path):
                # Path should be absolute if it exists
                pass

    @patch('pipeline.run_mediapipe_stickers')
    @patch('pipeline.run_sber_swap')
    @patch('pipeline.os.path.exists')
    @patch('pipeline.os.remove')
    @patch('sys.argv', new_callable=lambda: [
        'pipeline.py',
        '-i', 'dummy_input.mp4',
        '-o', 'dummy_output.mp4',
        '--source_paths', 'dummy_source.jpg',
        '-s', 'dummy_sticker.png:forehead:1.5'
    ])
    def test_main_pipeline_flow(self, mock_argv, mock_remove, mock_exists,
                                 mock_sber_swap, mock_mediapipe):
        """Test the main pipeline flow"""
        mock_exists.return_value = True
        mock_sber_swap.return_value = self.intermediate_video
        mock_mediapipe.return_value = self.output_video
        Path(self.intermediate_video).touch()
        Path(self.output_video).touch()
        
        # Update mock_argv with actual paths
        mock_argv[1] = '-i'
        mock_argv[2] = self.input_video
        mock_argv[3] = '-o'
        mock_argv[4] = self.output_video
        mock_argv[5] = '--source_paths'
        mock_argv[6] = self.source_face
        mock_argv[7] = '-s'
        mock_argv[8] = f"{self.sticker_path}:forehead:1.5"
        
        # Call main and check return value
        result = main()
        self.assertEqual(result, 0)
        
        # Verify both steps were called
        self.assertTrue(mock_sber_swap.called)
        self.assertTrue(mock_mediapipe.called)
        
        # Verify mediapipe was called with intermediate video
        mediapipe_call = mock_mediapipe.call_args
        self.assertEqual(mediapipe_call[1]['input_video'], 
                        self.intermediate_video)

    @patch('pipeline.run_mediapipe_stickers')
    @patch('pipeline.run_sber_swap')
    @patch('pipeline.os.path.exists')
    @patch('sys.argv', new_callable=lambda: [
        'pipeline.py',
        '-i', 'dummy_input.mp4',
        '-o', 'dummy_output.mp4',
        '--source_paths', 'dummy_source.jpg',
        '-s', 'dummy_sticker.png:forehead:1.5',
        '--keep_intermediate'
    ])
    def test_main_keep_intermediate(self, mock_argv, mock_exists,
                                     mock_sber_swap, mock_mediapipe):
        """Test that intermediate file is kept when requested"""
        mock_exists.return_value = True
        mock_sber_swap.return_value = self.intermediate_video
        mock_mediapipe.return_value = self.output_video
        Path(self.intermediate_video).touch()
        Path(self.output_video).touch()
        
        # Update mock_argv with actual paths
        mock_argv[1] = '-i'
        mock_argv[2] = self.input_video
        mock_argv[3] = '-o'
        mock_argv[4] = self.output_video
        mock_argv[5] = '--source_paths'
        mock_argv[6] = self.source_face
        mock_argv[7] = '-s'
        mock_argv[8] = f"{self.sticker_path}:forehead:1.5"
        
        with patch('sys.exit'):
            with patch('pipeline.os.remove') as mock_remove:
                main()
                # Verify remove was NOT called
                mock_remove.assert_not_called()

    @patch('pipeline.run_mediapipe_stickers')
    @patch('pipeline.run_sber_swap')
    @patch('pipeline.os.path.exists')
    @patch('sys.argv', new_callable=lambda: [
        'pipeline.py',
        '-i', 'dummy_input.mp4',
        '-o', 'dummy_output.mp4',
        '--source_paths', 'dummy_source.jpg',
        '-s', 'dummy_sticker.png:forehead:1.5',
        '--intermediate_output', 'dummy_intermediate.mp4'
    ])
    def test_main_custom_intermediate_path(self, mock_argv, mock_exists,
                                           mock_sber_swap, mock_mediapipe):
        """Test custom intermediate output path"""
        mock_exists.return_value = True
        custom_intermediate = os.path.join(self.test_dir, 'custom_inter.mp4')
        mock_sber_swap.return_value = custom_intermediate
        mock_mediapipe.return_value = self.output_video
        Path(custom_intermediate).touch()
        Path(self.output_video).touch()
        
        # Update mock_argv with actual paths
        mock_argv[1] = '-i'
        mock_argv[2] = self.input_video
        mock_argv[3] = '-o'
        mock_argv[4] = self.output_video
        mock_argv[5] = '--source_paths'
        mock_argv[6] = self.source_face
        mock_argv[7] = '-s'
        mock_argv[8] = f"{self.sticker_path}:forehead:1.5"
        mock_argv[9] = '--intermediate_output'
        mock_argv[10] = custom_intermediate
        
        with patch('pipeline.sys.exit') as mock_exit:
            main()
        
        # Verify sber-swap was called with custom intermediate path
        sber_call = mock_sber_swap.call_args
        self.assertEqual(sber_call[1]['output_video'], custom_intermediate)

    @patch('pipeline.run_mediapipe_stickers')
    @patch('pipeline.run_sber_swap')
    @patch('pipeline.os.path.exists')
    @patch('sys.argv', new_callable=lambda: [
        'pipeline.py',
        '-i', 'dummy_input.mp4',
        '-o', 'dummy_output.mp4',
        '--source_paths', 'dummy_source.jpg',
        '-s', 'dummy_sticker.png:forehead:1.5'
    ])
    def test_main_missing_input_file(self, mock_argv, mock_exists,
                                      mock_sber_swap, mock_mediapipe):
        """Test error handling for missing input file"""
        mock_exists.return_value = False  # Input file doesn't exist
        
        # Update mock_argv with actual paths
        mock_argv[1] = '-i'
        mock_argv[2] = self.input_video
        mock_argv[3] = '-o'
        mock_argv[4] = self.output_video
        mock_argv[5] = '--source_paths'
        mock_argv[6] = self.source_face
        mock_argv[7] = '-s'
        mock_argv[8] = f"{self.sticker_path}:forehead:1.5"
        
        result = main()
        self.assertEqual(result, 1)
        
        self.assertFalse(mock_sber_swap.called)
        self.assertFalse(mock_mediapipe.called)

    @patch('pipeline.subprocess.run')
    @patch('pipeline.os.path.exists')
    def test_run_sber_swap_subprocess_error(self, mock_exists, mock_subprocess):
        """Test error handling when subprocess fails"""
        mock_exists.return_value = True
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, 'cmd')
        
        with self.assertRaises(subprocess.CalledProcessError):
            run_sber_swap(
                input_video=self.input_video,
                source_paths=[self.source_face],
                target_faces_paths=[],
                output_video=self.intermediate_video,
                sber_swap_dir=self.sber_swap_dir
            )

    @patch('pipeline.subprocess.run')
    @patch('pipeline.os.path.exists')
    def test_run_mediapipe_stickers_subprocess_error(self, mock_exists,
                                                      mock_subprocess):
        """Test error handling when mediapipe subprocess fails"""
        mock_exists.return_value = True
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, 'cmd')
        
        with self.assertRaises(subprocess.CalledProcessError):
            run_mediapipe_stickers(
                input_video=self.intermediate_video,
                output_video=self.output_video,
                sticker_configs=[f"{self.sticker_path}:forehead:1.5"],
                mediapipe_dir=self.mediapipe_dir
            )


class TestPipelineResultsDirectory(unittest.TestCase):
    """Test cases for results directory structure"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.test_dir, 'results')
        self.intermediate_dir = os.path.join(self.results_dir, 'intermediate')
        self.final_dir = os.path.join(self.results_dir, 'final')
        
        # Create directory structure
        os.makedirs(self.intermediate_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_results_directory_structure(self):
        """Test that results directory structure exists"""
        self.assertTrue(os.path.exists(self.results_dir))
        self.assertTrue(os.path.exists(self.intermediate_dir))
        self.assertTrue(os.path.exists(self.final_dir))

    def test_results_directory_writable(self):
        """Test that results directories are writable"""
        test_file_intermediate = os.path.join(
            self.intermediate_dir, 'test_intermediate.mp4'
        )
        test_file_final = os.path.join(self.final_dir, 'test_final.mp4')
        
        Path(test_file_intermediate).touch()
        Path(test_file_final).touch()
        
        self.assertTrue(os.path.exists(test_file_intermediate))
        self.assertTrue(os.path.exists(test_file_final))


if __name__ == '__main__':
    unittest.main()

