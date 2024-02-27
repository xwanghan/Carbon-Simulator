import json
import lz4.frame

filepath = "0/flat/dense_logs/logs_00/"
f1 = filepath+"env-02.lz4"
compression_level = 16

with lz4.frame.open(
        f1, mode="rb", compression_level=compression_level
) as log_file:
    log = json.loads(bytes.decode(log_file.read()))


print(log)
