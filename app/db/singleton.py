import threading
import chromadb

class ChromaDBSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, path="./chroma_db"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_client(path)
        return cls._instance

    def _init_client(self, path):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection("cultural_stories")

    def get_collection(self):
        return self.collection