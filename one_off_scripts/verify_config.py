import yaml
with open("config/model_config.yaml") as f:
    data = yaml.safe_load(f)
keys = list(data.keys())
print(f"Total configs: {len(keys)}")
single = [k for k in keys if "SingleTask" in k]
print(f"SingleTask configs: {single}")
# Check one
if "SingleTask_click" in data:
    c = data["SingleTask_click"]
    print(f"num_tasks: {c.get("num_tasks")}")
    print(f"task_labels: {c.get("task_labels")}")
    print(f"model: {c.get("model")}")
