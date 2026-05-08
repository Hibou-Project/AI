from abc import ABC, abstractmethod


class BaseProvider(ABC):

    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def get_dataset_dir(self):
        pass

    # @abstractmethod
    # def save(self, save_dir: str, image_transform: str):
    #     pass