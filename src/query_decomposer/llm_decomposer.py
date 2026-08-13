import json
import time
from loguru import logger
from groq import Groq, RateLimitError

from schemas.schemas import DecomposeResult
from config_loader import load_appcfg, load_prompts_cfg

class LLMDecomposer:
    """Groq API kullanarak karmaşık sorguları alt sorgulara bölen sınıf."""
    
    def __init__(self, model: str | None = None, max_sub_queries: int | None = None):
        app_cfg = load_appcfg()
        decomposer_cfg = getattr(app_cfg, "decomposer", {})
        prompts_cfg = load_prompts_cfg()
        
        self.model = model or decomposer_cfg.get("model", "qwen/qwen3.6-27b")
        self.max_sub_queries = max_sub_queries or decomposer_cfg.get("max_sub_queries", 3)
        self.min_sub_queries = decomposer_cfg.get("min_sub_queries", 2)
        self.temperature = decomposer_cfg.get("temperature", 0.1)
        self.max_tokens = decomposer_cfg.get("max_tokens", 1024)
        
        decomposer_prompt_cfg = getattr(prompts_cfg, "decomposer_prompt", {}) or {}

        self.system_prompt_template = decomposer_prompt_cfg.get(
            "system",
            "Verilen karmaşık soruyu maksimum {max_sub_queries} bağımsız alt sorguya böl.\n"
            "Yanıtını yalnızca JSON olarak ver: {\"sub_queries\": [\"...\"], \"reasoning\": \"...\"}\n",
        )

        self.client = Groq()

    def decompose(self, query: str) -> DecomposeResult:
        """Sorguyu LLM kullanarak alt sorgulara böler."""
        system_prompt = self.system_prompt_template.replace("{max_sub_queries}", str(self.max_sub_queries))
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                data = json.loads(content)
                
                sub_queries = data.get("sub_queries", [])
                reasoning = data.get("reasoning", "")
                
                if len(sub_queries) < self.min_sub_queries:
                    logger.warning(f"Yetersiz alt soru ({len(sub_queries)}). Fallback uygulanıyor.")
                    return DecomposeResult(
                        sub_queries=[], 
                        reasoning=f"fallback: yetersiz alt soru. {reasoning}"
                    )
                
                if len(sub_queries) > self.max_sub_queries + 1:
                    logger.warning(
                        f"Aşırı alt soru üretildi ({len(sub_queries)} > {self.max_sub_queries + 1}). "
                        "Simple retrieval'a fallback uygulanıyor."
                    )
                    return DecomposeResult(
                        sub_queries=[],
                        reasoning=f"fallback: aşırı alt soru üretimi ({len(sub_queries)}). {reasoning}"
                    )

                if len(sub_queries) > self.max_sub_queries:
                    logger.warning(
                        f"Alt soru sayısı ({len(sub_queries)}) sınırı aştı. "
                        f"İlk {self.max_sub_queries} soru alınıyor."
                    )
                    sub_queries = sub_queries[:self.max_sub_queries]
                    
                return DecomposeResult(sub_queries=sub_queries, reasoning=reasoning)
                
            except RateLimitError as e:
                sleep_time = 1.5 * (2 ** attempt)
                logger.warning(
                    f"Groq Rate Limit hatası (Deneme {attempt + 1}/3). "
                    f"{sleep_time:.1f}sn üssel bekleniyor... Hata: {e}"
                )
                if attempt < 2:
                    time.sleep(sleep_time)
                else:
                    logger.error("Maksimum rate limit denemesi aşıldı.")
                    return DecomposeResult(
                        sub_queries=[],
                        reasoning=f"fallback: rate limit aşıldı. {e}",
                    )
            except Exception as e:
                logger.error(f"LLM decomposition hatası: {e}")
                return DecomposeResult(
                    sub_queries=[], 
                    reasoning=f"fallback: api/parse hatası. {e}"
                )
                
        return DecomposeResult(sub_queries=[], reasoning="fallback: bilinmeyen hata")
