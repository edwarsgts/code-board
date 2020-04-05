# Working with timestamps
import time
from datetime import datetime, timedelta

# def send_emails():
#     for i in range(160000):
#         pass

# start = time.time()
# send_emails()
# end = time.time()
# print(end-start)

dt1 = datetime(2020, 4, 3) + timedelta(days=1)
dt2 = datetime.now()
dt3 = datetime.strptime("2018/01/01", "%Y/%m/%d")
dt4 = datetime.fromtimestamp(time.time())


# print(dt4)
# print(f"{dt4.year}/{dt4.month}")
# dt4.strftime("%Y/%m")

# print(dt4)
# Working with time deltas

duration = dt4 - dt3
print(duration)
print(duration.days)
print(duration.total_seconds())
