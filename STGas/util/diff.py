# 多帧差
def multi_frame_diff(frame_list):
    frame_diffs = []
    for interval in range(len(frame_list) - 1):
        for i in range(len(frame_list) - interval - 1):
            frame_diffs.append(frame_list[i + interval + 1] - frame_list[i])
    return frame_diffs
