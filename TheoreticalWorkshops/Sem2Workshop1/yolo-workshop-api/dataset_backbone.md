
# Overview

This file contains a class that will be used for storing datasets so that they can interact with the PyTorch Framework

Some of the code has been deliberately left for completion in the Kaggle notebook.



# Key Components of a Dataset Class

- Must be a child class of torch.utils.data.Dataset
- Every instance store the file paths of the data as well as labels
- Must have a `__len__()` implementation
- Must have a `__getitem()__` implementation



# Why torch.utils.data.Dataset?

You can technically just store your data as a numpy array and loop through that, however, if you do this would need to implementing features such as batching yourself.
The issue is that the DataLoader class that is part of the torch library expects and object that follows a certain structure. The reason we use DataLoader is that it provides the batching functions, etc. that are very useful.

If our dataset object is a child class of the Dataset class then it will be perfectly compatible with the DataLoader constructor.



# Storing a reference to the data within the Dataset object

You can really do this any way you like so long as you store a way of getting these for each data point:

- the labels for the data  (if doing supervised learning)
- the paths to the images (if doing computer vision)4
- anything else along those lines - it's obviously project dependent

However, one of the best ways to store the data (in my opinion) is by storing all of the labels as a pandas dataframe and just having the dataset class instance have a variable storing that dataframe.

This can make your constructor super simple, e.g.

```
def __init__(self, dataframe):
	self.df = dataframe
```



### transform argument

If we need to convert data from e.g. an image, we can use the transform library in torch.

If we're transforming the data we need to pass this in as an argument in the constructor

e.g.

```
def __init__(self, dataframe, transform=None):
	self.df = dataframe
	self.transform = transform
```

*Note: in the parameters we set transform=None to make it an optional argument. If the constructor receives no argument for transform, it defaults to None*


We then define the behaviour of transform within main.py 

e.g.

```
# 1. Define the "Recipe" (The Transform)
my_recipe = transforms.Compose([
    transforms.Resize((224, 224)),  # defines the transformation to the tensor
    transforms.ToTensor()
])

# 2. Give the recipe to the Dataset constructor
# This is where 'my_recipe' becomes 'self.transform' inside the class
train_data = RaccoonDataset(dataframe=df, transform=my_recipe)


```

We can then go and use the transform value in the `__getitem__()` function to manipulate our tensor input to the model.

# The `__len()__` function

The DataLoader constructor requires the input object to have a function for getting the number of training examples in your dataset.

If your storing your data as a pandas dataframe, your implementation can literally be as simple as this:

```
def __len__(self):
	return len(self.df)
```



# The `__getitem__()` function

 The DataLoader constructor also requires that the argument object must have a function for getting the datapoint at a given row in the dataset.

==The datapoint vector must be a torch tensor, i.e. the `getitem` function must return a tuple containing (the truth label of the point as a tensor, feature vector as a tensor)==
 
*Note: this implementation assumes that you're using pandas for storing your data*

```
def __getitem__(self, idx):
	# Use .iloc to ensure we get the row by position, not by index label
	row = self.df.iloc[idx]
	
	# Convert to numpy first, then to a torch Tensor
	point = torch.tensor(row[['feature1','feature2','feature3']].values).float()
	label = torch.tensor(row['target']).long()
	
	return features, label  # MUST BE IN THIS ORDER
```


**Why to numpy array first?**

- **The Problem:** A Pandas Series contains metadata (index labels, data types, etc.) that PyTorch doesn't know how to handle.

- **The Solution:** Converting to numpy via `.values` or `.to_numpy()` strips away the "Pandas fluff," leaving behind a clean, raw block of memory that PyTorch can instantly adopt.




### What if my data is images or something else?

==You alter the `__getitem__` implementation if this is the case. You still return the same type of tuple though, i.e. (the truth label of the point as a tensor, feature vector as a tensor)==

e.g.

```
def __getitem__(self, idx):
        # 1. Get the specific file path for this index
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 2. Load the image from disk
        # We use .convert("RGB") to ensure 3 channels (handles grayscale/RGBA)
        image = Image.open(img_path).convert("RGB") 
        
        # 3. Get the corresponding label
        label = self.labels[idx]

        # 4. Transform: This is where Resize and ToTensor() usually happen
        if self.transform:
            image = self.transform(image)

        # 5. Return the "Input-Target" pair
        return image, label
```