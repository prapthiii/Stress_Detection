import os

dataset_path = r"E:\Stress management\ravdess-emotional-speech-audio"
actors = os.listdir(dataset_path)
print("Total actor folders:", len(actors))

sample_actor = os.path.join(dataset_path, actors[0])
files = os.listdir(sample_actor)

print("Sample files:", files[:5])