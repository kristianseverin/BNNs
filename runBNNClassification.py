from utils import Config


cuda = torch.cuda.is_available()
print("CUDA Available: ", cuda)

if cuda:
    gpu = GPUtil.getFirstAvailable()
    print("GPU Available: ", gpu)
    torch.cuda.set_device(gpu)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("Device: ", device)

# read data and preprocess
df = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_int.csv')