# ML-Final

This is the final project for the machine learning course.

# How to Start

1. Prepare Data

   ```shell
   cd ML-Final
   unzip data/data.zip
   ```

   Make sure that data path is `ML-Final/data/data_96-xx`
2. Configure Settings

   Copy `ML-Final/src/run.py` to do experiments. ** **Avoid modifying the original file!** **

   Open `run.py`. Locate **TODO** marks. Modify as needed.
3. Run Code

   ```shell
   python run.py
   ```

## File Structure

```
ML-Final
|-README.md
|-requirements.txt
|-data
| |-raw_data
| |-notes
| | |-数据分析.md
|-references
| |-references code
| |-references note
| |-references paper
|-requirements.txt
|-src
| |-analyze.py
| |-arguments.py
| |-data.py
| |-gpuManager.py
| |-logger.py
| |-main.py
| |-metrics.py
| |-model
| | |-__init__.py
| | |-_test_model.py
| | |-configs.py
| | |-criterion.py
| | |-LSTM.py
| | |-Transformer.py
| | |-GRU.py
| | |-CNN.py
| | |-ExtendedVecto.py
| |-run.py
| |-utils.py
|-experiment
| |-ckpt
| |-experiment_analysis
| | |-visualize.ipynb
| |-log_space
```
