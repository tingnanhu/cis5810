# Pipeline Tests

This directory contains tests for the pipeline script that chains sber-swap face swap and mediapipe sticker overlay.

## Running Tests

Run all tests:
```bash
python -m pytest test/test_pipeline.py -v
```

Or using unittest:
```bash
python -m unittest test.test_pipeline -v
```

Run a specific test class:
```bash
python -m unittest test.test_pipeline.TestPipeline -v
```

## Test Structure

### TestPipeline
Tests for the main pipeline functionality:
- `test_run_sber_swap_basic`: Basic sber-swap execution
- `test_run_sber_swap_with_options`: sber-swap with optional parameters
- `test_run_sber_swap_absolute_paths`: Path conversion to absolute
- `test_run_sber_swap_missing_script`: Error handling for missing script
- `test_run_mediapipe_stickers_basic`: Basic mediapipe execution
- `test_run_mediapipe_stickers_multiple`: Multiple stickers
- `test_run_mediapipe_stickers_absolute_paths`: Path conversion
- `test_main_pipeline_flow`: Full pipeline flow
- `test_main_keep_intermediate`: Keep intermediate file option
- `test_main_custom_intermediate_path`: Custom intermediate path
- `test_main_missing_input_file`: Error handling for missing input
- `test_run_sber_swap_subprocess_error`: Subprocess error handling
- `test_run_mediapipe_stickers_subprocess_error`: Subprocess error handling

### TestPipelineResultsDirectory
Tests for results directory structure:
- `test_results_directory_structure`: Verify directory structure exists
- `test_results_directory_writable`: Verify directories are writable

## Test Assumptions

The tests use mocking to avoid actually running sber-swap and mediapipe:
- `subprocess.run` is mocked to simulate successful/failed executions
- File system operations are mocked where appropriate
- Tests assume sber-swap will produce a face-swapped video (mocked)
- Tests assume mediapipe will add stickers successfully (mocked)

## Results Directory

The `results/` directory structure:
```
test/results/
├── intermediate/    # Face-swapped videos (before stickers)
└── final/          # Final videos (with face swap and stickers)
```

This structure is automatically created when using `--use_results_dir` flag.

