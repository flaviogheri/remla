import time

# Simulate some time-consuming task
def simulate_task():
    print("Starting time-consuming task...")
    time.sleep(5)  # Simulate a task that takes 5 seconds
    print("Time-consuming task completed.")

if __name__ == "__main__":
    simulate_task()