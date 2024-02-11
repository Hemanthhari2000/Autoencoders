import matplotlib.pyplot as plt
import torch

METRIC_PATH = './data/metrics/1706861588'

losses = torch.load(f'{METRIC_PATH}/losses.pt')
output = torch.load(f'{METRIC_PATH}/outputs.pt')


plt.style.use('fivethirtyeight')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.plot(losses)
# plt.show()

_, image, reconstructed_image = output[0]

plt.imshow(image[0].reshape(28, 28))
plt.imshow(reconstructed_image[0].detach().numpy().reshape(28, 28))
plt.show()