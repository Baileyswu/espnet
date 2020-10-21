import json
import matplotlib.pyplot as plt
import kaldiio

# load 10-th speech/text in data.json
root = "/home/dingchaoyue/speech/dysarthria/espnet/egs/an4/asr1"
with open(root + "/dump/test/deltafalse/data.json", "r") as f:
  test_json = json.load(f)["utts"]

key, info = list(test_json.items())[10]

# plot the speech feature
fbank = kaldiio.load_mat(info["input"][0]["feat"])
plt.matshow(fbank.T[::-1])
plt.title(key + ": " + info["output"][0]["text"])
print("Hello!")
plt.savefig("/home/dingchaoyue/speech/dysarthria/espnet/test.png",dpi=500)

# print the key-value pair
key, info


