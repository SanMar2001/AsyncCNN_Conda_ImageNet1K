import numpy as np
import torch
from time import time
from torch import nn
import pickle, asyncio
from torch.optim import SGD
from torch.utils.data import Subset, DataLoader
import os

MSG_INIT = "INIT"
MSG_NEXT_BATCH = "NEXT_BATCH"
MSG_GRADIENTS = "GRADIENTS"
MSG_TRAINING_DONE = "TRAINING_DONE"
MSG_WORKER_DONE = "WORKER_DONE"

import numpy as np
import torch
from time import time
from torch import nn
import pickle, asyncio
from torch.optim import SGD
from torch.utils.data import Subset, DataLoader
import os

MSG_INIT = "INIT"
MSG_NEXT_BATCH = "NEXT_BATCH"
MSG_GRADIENTS = "GRADIENTS"
MSG_TRAINING_DONE = "TRAINING_DONE"
MSG_WORKER_DONE = "WORKER_DONE"

class ParameterServer:
    def __init__(self, id, host, port, model, longitude,
                 epochs=10, batch_size=128, lr=0.08, alpha=1.0):
        self.id = id
        self.host = host
        self.port = port
        self.model = model
        self.longitude = longitude
        self.indexes = np.random.permutation(longitude).tolist()
        self.batch_size = batch_size
        self.batch_pointer = 0
        self.base_lr = lr
        self.alpha = alpha
        self.optimizer = SGD(self.model.parameters(), lr, momentum=0.0)
        self.epochs = epochs
        self.current_epoch = 0
        self.server = None
        self.connections = {}
        self.model_version = 0
        self.time = None
        self.lock = asyncio.Lock()
        self.active_workers = 0

    async def start(self):
        self.server = await asyncio.start_server(self.handle_worker, self.host, self.port)
        os.system("clear")
        print(f"Parameter Server running on {self.host}:{self.port}")
        try:
            await self.server.serve_forever()
        except asyncio.CancelledError:
            print("Server stopped cleanly.")

    def get_next_batch(self):
        start = self.batch_pointer
        end = min(start + self.batch_size, self.longitude)
        if start >= self.longitude:
            return None
        self.batch_pointer = end
        return start, end

    def reset_epoch(self):
        self.batch_pointer = 0
        self.indexes = np.random.permutation(self.longitude).tolist()
        self.current_epoch += 1

    async def handle_worker(self, reader, writer):
        addr = writer.get_extra_info("peername")
        self.connections[addr] = writer
        self.active_workers += 1
        print(f"Worker connected: {addr} | Active workers: {self.active_workers}")

        try:
            init_content = {
                "type": MSG_INIT,
                "state_dict": self.model.state_dict(),
                "indexes": self.indexes,
                "batch_range": self.get_next_batch(),
                "model_version": self.model_version
            }
            self.time = time()
            await self.send_payload(writer, init_content)

            while True:
                try:
                    message = await self.recv_payload(reader)
                except asyncio.IncompleteReadError:
                    print(f"Worker {addr} disconnected unexpectedly.")
                    break

                msg_type = message.get("type")

                if msg_type == MSG_NEXT_BATCH:
                    batch_range = self.get_next_batch()
                    if batch_range is None:
                        epoch_time = time() - self.time
                        print(f"Finished epoch {self.current_epoch} | Time: {epoch_time:.2f}s")
                        if self.current_epoch >= self.epochs:
                            await self.send_payload(writer, {"type": MSG_TRAINING_DONE})
                            break
                        self.reset_epoch()
                        self.time = time()
                        batch_range = self.get_next_batch()

                    await self.send_payload(writer, {
                        "type": MSG_NEXT_BATCH,
                        "batch_range": batch_range,
                        "state_dict": self.model.state_dict(),
                        "model_version": self.model_version
                    })

                elif msg_type == MSG_GRADIENTS:
                    worker_version = message.get("model_version", 0)
                    grads = message.get("data")
                    staleness = max(0, self.model_version - worker_version)
                    async_lr = self.base_lr / (1.0 + self.alpha * staleness)

                    async with self.lock:
                        self.optimizer.param_groups[0]['lr'] = async_lr
                        self.optimizer.zero_grad()
                        for param, grad in zip(self.model.parameters(), grads):
                            param.grad = grad.clone().to(param.device)
                        self.optimizer.step()
                        self.model_version += 1

                    batch_range = self.get_next_batch()
                    if batch_range is None:
                        epoch_time = time() - self.time
                        print(f"Finished epoch {(self.current_epoch + 1)} | Time: {epoch_time:.2f}s")
                        if self.current_epoch >= self.epochs:
                            await self.send_payload(writer, {"type": MSG_TRAINING_DONE})
                            break
                        self.reset_epoch()
                        self.time = time()
                        batch_range = self.get_next_batch()

                    await self.send_payload(writer, {
                        "type": MSG_NEXT_BATCH,
                        "batch_range": batch_range,
                        "state_dict": self.model.state_dict(),
                        "model_version": self.model_version
                    })

                elif msg_type == MSG_WORKER_DONE:
                    print(f"Worker {addr} finished training.")
                    break

                else:
                    print(f"Unknown message type from {addr}: {msg_type}")
                    break

        except (ConnectionResetError, BrokenPipeError):
            print(f"Connection error with worker {addr}.")
        finally:
            self.active_workers -= 1
            print(f"Connection closed: {addr} | Active workers: {self.active_workers}")
            self.connections.pop(addr, None)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

            if self.active_workers == 0:
                print("All workers finished. Shutting down server...")
                self.server.close()
                await self.server.wait_closed()
                for task in asyncio.all_tasks():
                    if task is not asyncio.current_task():
                        task.cancel()

    async def send_payload(self, writer, obj):
        data = pickle.dumps(obj)
        writer.write(len(data).to_bytes(4, "big") + data)
        try:
            await writer.drain()
        except BrokenPipeError:
            pass

    async def recv_payload(self, reader):
        msg_len_data = await reader.readexactly(4)
        msg_len = int.from_bytes(msg_len_data, "big")
        data = await reader.readexactly(msg_len)
        return pickle.loads(data)

    
class Worker:
    def __init__(self, dataset, model, host, port, device=None):
        self.dataset = dataset
        self.model = model()
        self.host = host
        self.port = port
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model_version = 0

    async def run(self):
        await asyncio.sleep(2)
        try:
            reader, writer = await asyncio.open_connection(self.host, self.port)
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return

        try:
            init_msg = await self.recv_payload(reader)
            self.model.load_state_dict(init_msg["state_dict"])
            indexes = init_msg["indexes"]
            batch_range = init_msg["batch_range"]
            self.model_version = init_msg.get("model_version", 0)
            loss_fn = nn.CrossEntropyLoss()

            while True:
                start, end = batch_range
                if start is None or end is None:
                    break

                batch_idx = indexes[start:end]
                subset = Subset(self.dataset, batch_idx)
                loader = DataLoader(subset, batch_size=128, shuffle=False, pin_memory=True, num_workers=2)
                self.model.train()
                self.model.zero_grad()

                for x_batch, y_batch in loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    y_pred = self.model(x_batch)
                    loss = loss_fn(y_pred, y_batch)
                    loss.backward()

                grads = [param.grad.clone().cpu() for param in self.model.parameters()]

                await self.send_payload(writer, {
                    "type": MSG_GRADIENTS,
                    "data": grads,
                    "model_version": self.model_version
                })

                next_msg = await self.recv_payload(reader)

                if next_msg.get("type") == MSG_TRAINING_DONE:
                    await self.send_payload(writer, {"type": MSG_WORKER_DONE})
                    print("Training finished. Closing worker")
                    return

                if "state_dict" in next_msg:
                    self.model.load_state_dict(next_msg["state_dict"])
                    self.model.to(self.device)
                    self.model_version = next_msg.get("model_version", self.model_version)

                batch_range = next_msg.get("batch_range", (None, None))

        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError) as e:
            print(f"Connection lost: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except BrokenPipeError:
                pass

    async def send_payload(self, writer, obj):
        data = pickle.dumps(obj)
        writer.write(len(data).to_bytes(4, "big") + data)
        await writer.drain()

    async def recv_payload(self, reader):
        msg_len = int.from_bytes(await reader.readexactly(4), "big")
        data = await reader.readexactly(msg_len)
        return pickle.loads(data)

#ImageNet1K Model
class Convolutional_ImageNet(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), 
            nn.Linear(4096, 1024), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(1024, 512), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
#Food101 Model
class Convolutional_Food101(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), 
            nn.Linear(4096, 1024), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(1024, 512), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

#CIFAR10 Model
class Convolutional_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
