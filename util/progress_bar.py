import sys


# 进度条
def load_data_progress_bar(total, progress):
    """
    进度条
    :param total: 进度条的总长度（或总任务量）
    :param progress: 当前进度（已完成的任务量）
    """
    bar_length = 50
    filled_length = int(round(bar_length * progress / float(total)))
    # percents = round(100.0 * progress / float(total), 1)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    end_word = "\n" if total == progress else ""
    sys.stdout.write(f'\rLoad Data {progress}/{total}: [{bar}]{end_word}')
    sys.stdout.flush()
