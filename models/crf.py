import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))

    def forward(self, emissions, tags, mask):
        log_denominator = self._compute_log_partition(emissions, mask)
        log_numerator = self._score_sentence(emissions, tags, mask)
        return torch.mean(log_denominator - log_numerator)

    def _compute_log_partition(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.size()
        score = self.start_transitions + emissions[:, 0]
        for t in range(1, seq_len):
            emit_t = emissions[:, t].unsqueeze(2)
            trans = self.transitions.unsqueeze(0)
            score_t = score.unsqueeze(1) + trans + emit_t
            score_t = torch.logsumexp(score_t, dim=2)
            score = torch.where(mask[:, t].unsqueeze(1), score_t, score)
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _score_sentence(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.size()
        score = self.start_transitions[tags[:, 0]]
        score += emissions[torch.arange(batch_size), 0, tags[:, 0]]
        for t in range(1, seq_len):
            trans_score = self.transitions[tags[:, t - 1], tags[:, t]]
            emit_score = emissions[torch.arange(batch_size), t, tags[:, t]]
            score += (trans_score + emit_score) * mask[:, t]
        last_tag_index = mask.long().sum(dim=1) - 1
        last_tags = tags[torch.arange(batch_size), last_tag_index]
        score += self.end_transitions[last_tags]
        return score

    def decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.size()
        score = self.start_transitions + emissions[:, 0]
        history = []
        for t in range(1, seq_len):
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_path = next_score.max(dim=1)
            best_score = best_score + emissions[:, t]
            score = torch.where(mask[:, t].unsqueeze(1), best_score, score)
            history.append(best_path)
        score += self.end_transitions
        best_last_score, best_last_tag = score.max(dim=1)
        sequences = []
        for i in range(batch_size):
            seq_end = int(mask[i].sum().item())
            tag = best_last_tag[i].item()
            best_tags = [tag]
            for hist_t in reversed(history[:seq_end - 1]):
                tag = hist_t[i][tag].item()
                best_tags.append(tag)
            sequences.append(list(reversed(best_tags)))
        return sequences
