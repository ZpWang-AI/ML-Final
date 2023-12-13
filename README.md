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

   In `ML-Final/src/run.py`. Marked with **TODO**.
3. Run Code

   ```shell
   python run.py
   ```

## Contributions

Refer to the **Issues** and **TODO.md**.

## File Structure

```
ML-Final
|-TODO.md
|-README.md
|-data
| |-data.ipynb
| |-data.zip
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
| | |-configs.py
| | |-criterion.py
| |-run.py
| |-utils.py
|-experiment
| |-ckpt
| |-experiment_analysis
| | |-visualize.ipynb
| |-log_space
```
