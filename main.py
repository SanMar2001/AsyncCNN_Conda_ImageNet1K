import asyncio
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from models import Convolutional_Food101, Worker, ParameterServer
import torch

def main(args):
    if args.role == "parameter_server":
        p_server = ParameterServer(
            id = args.id,
            host = args.host,
            port = args.port,
            model = args.model,
            longitude = args.longitude
        )
        asyncio.run(p_server.start())
    
    elif args.role == "worker":
        worker = Worker(
            dataset = args.dataset,
            model = args.classmodel,
            host = args.host,
            port = args.port
        )
        time.sleep(3)
        asyncio.run(worker.run())

if __name__ == "__main__":
    #Loading arguments from console
    parser = argparse.ArgumentParser(description="Distributed Training")
    parser.add_argument('--role', choices=["parameter_server", "worker"],
                        required=True)
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--host', type=str)
    parser.add_argument('--port', type=int)
    
    #Transformations for the Dataset
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    #Downloading Dataset and spliting in train and test
    train_dataset = datasets.Food101(
        root="data",
        split="train",
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.Food101(
        root="data",
        split="test",
        download=False,
        transform=test_transform
    )
    print("Dataset downloaded or already founded")
    longitude = len(train_dataset)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Convolutional_Food101(num_classes=101)
    model.to(device)
    parser.set_defaults(model=model, classmodel=Convolutional_Food101,
                        dataset=train_dataset, longitude=longitude)
    
    args = parser.parse_args()
    main(args)
    if args.role == "parameter_server":
        model.eval()
        test_loader = DataLoader(test_dataset, batch_size=128,
                                 shuffle=False, num_workers=4, pin_memory=True)
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        
        print(f"Final accuracy in test: {((correct/total)*100):.2f}%")

    