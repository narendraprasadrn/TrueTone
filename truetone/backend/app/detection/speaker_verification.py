import os
from typing import Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier

class SpeakerVerifier:
    def __init__(self, savedir: str = "pretrained_models/ecapa-tdnn", device: str = "cpu"):
        self.device = torch.device(device)
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            run_opts={"device": device}
        )
        self.classifier.eval()
        
        # In-memory storage for enrolled embeddings.
        # NOTE: In production, these should be persisted encrypted at rest.
        self.enrollments: Dict[str, torch.Tensor] = {}

    def _extract_embedding(self, audio: np.ndarray) -> torch.Tensor:
        # Convert to tensor and shape [1, time]
        signal = torch.tensor(audio, dtype=torch.float32).to(self.device)
        if signal.ndim > 1:
            signal = signal.squeeze()
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
            
        with torch.no_grad():
            # SpeechBrain encodes and returns [batch, 1, embedding_dim]
            embeddings = self.classifier.encode_batch(signal)
            embeddings = embeddings.squeeze(1) # shape [batch, embedding_dim]
            
        return embeddings

    def enroll(self, identity_id: str, reference_audio: np.ndarray) -> None:
        """Extract + store embedding for this identity."""
        embedding = self._extract_embedding(reference_audio)
        self.enrollments[identity_id] = embedding.cpu()

    def score(self, window: np.ndarray, identity_id: str) -> Optional[float]:
        """Cosine similarity to enrolled embedding, 0-1 (1=same speaker).
        Returns None if identity_id has no enrollment — caller must treat
        None as 'skip this signal', not as 0."""
        if identity_id not in self.enrollments:
            return None
            
        reference_embedding = self.enrollments[identity_id].to(self.device)
        current_embedding = self._extract_embedding(window)
        
        # Cosine similarity between reference and current
        # F.cosine_similarity returns values in [-1, 1].
        # We can map [-1, 1] to [0, 1] using (sim + 1) / 2
        cos_sim = F.cosine_similarity(current_embedding, reference_embedding, dim=1)
        sim_val = cos_sim.item()
        
        normalized_sim = (sim_val + 1.0) / 2.0
        return float(normalized_sim)
