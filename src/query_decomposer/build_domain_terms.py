import json
import argparse
from pathlib import Path
from typing import Set

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

TR_ASCII_TRANS = str.maketrans('çğıöşü', 'cgiosu')
def normalize_turkish(text: str) -> str:
    """Normalize Turkish characters to ASCII."""
    if not text: 
        return ''
    text_lower = text.replace('İ', 'i').replace('I', 'ı').lower()
    return text_lower.translate(TR_ASCII_TRANS)

GENERIC_FILTER = {
    'var', 'olan', 'sonucu', 'durumunda', 'oldugunda', 'olusan',
    'neden', 'ortaya', 'cikan', 'meydana', 'gelen', 'degeri',
    'bicimde', 'nasil', 'etkileri', 'ozellikleri', 'temel',
    'cikti', 'gordugum', 'onceki', 'ayrica', 'sonrasinda',
    'degisimi', 'degisimleri', 'degisim', 'yuksek', 'dusuk',
    'buyuk', 'kucuk', 'orta', 'fazla', 'gore', 'dogru',
    'yanlis', 'dogrudan', 'dolayli', 'alt', 'ust', 'ilk',
    'son', 'once', 'sonra', 'esas', 'asil', 'tam', 'yarim',
    'sag', 'sol', 'ic', 'dis', 'on', 'arka', 'karsi', 'benzer',
    'karsilastir', 'karsilastirilmasi', 'karsilastirma',
    'eder', 'edilir', 'edilmis', 'olabilir', 'oldugunu',
    'olur', 'olusturdugu', 'olustugunda', 'olusum', 'verir',
    'gelir', 'geliyo', 'genel', 'hakkinda', 'iki', 'ikisi',
    'kesin', 'galiba', 'acaba', 'baksana', 'dediler', 'misin',
    'misiniz', 'demek', 'anlama', 'anlatir', 'anlamadim',
    'anlasilir', 'cok', 'belirgin', 'bilgi', 'birbirinden',
    'bunun', 'ifade', 'tespit', 'tespiti', 'testi', 'tip',
    'tipi', 'sebebi', 'nedeni', 'nedenleri',
    'neticesinde', 'neye', 'neyi', 'bagla', 'kritik',
    'kullanilan', 'hesaplanir', 'nasildir', 'bagli',
    'arasindaki', 'sinyalde', 'sinyaldeki',
    'spektrumda', 'spektrumdaki', 'spektrumu', 'spektrumunda',
    'spektrumundaki', 'spektrumunun', 'gorülen', 'gorulur',
    'gosteriyor', 'gozlenen', 'orani', 'formunda', 'formundaki',
    'siniri', 'sinir', 'zone', 'iso', 'standardina', 'standart',
    'endüstriyel', 'endustriyel',
    'using', 'with', 'the', 'and', 'for', 'between',
    'what', 'how', 'type', 'are', 'you', 'can', 'key',
    'compare', 'differences', 'difference', 'differentiate',
    'measurements', 'dynamic', 'static', 'phase',
    'yoksa', 'veya', 'hem', 'ama', 'bir', 'kac', 'hangi',
    'ile', 'olarak', 'zaman', 'anda', 'ayni', 'biri',
    'cikar', 'degisir', 'nedir', 'nelerdir', 'midir',
    'anlarim', 'ederim', 'anlayamadim', 'titriyo',
    'gorulen', 'gosterir', 'dogrular', 'nedeniyle',
    'alinan', 'belirlenmistir', 'kabul', 'monte', 'uzerine',
    'uzeri', 'analizde', 'analiz', 'raporda', 'raporundaki',
    'problemi', 'yukselmis', 'kaynakli', 'yarattigi',
    'etkiler', 'etkisi', 'ayrilmasi', 'ayrilir',
    'gorseldeki', 'seste', 'edici', 'farklar', 'farki',
    'genligini', 'mekanizmasi', 'sayisi',
    '100', '300', '10816', '20816', '43x', '45x', '47x',
    'bolgesinde', 'kaide', 'rijit', 'esnek', 'dikey',
    'yatay', 'donel', 'fiziksel', 'matematiksel',
    'pompanin', 'pompalarda', 'motorlarinda', 'sistemlerinde',
    'milli', 'kovanli', 'rulmanli', 'yataklarda', 'kutularinda',
    'isareti', 'frekansli', 'frekanstaki',
    'bantlari', 'bantlarin', 'bantlarinin',
    'harmoniklerinin', 'titresimin',
}


def load_domain_terms(json_path: str) -> Set[str]:
    """
    Read the domain terms JSON file and return a set of terms.
    To be imported by classifier.py.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return set(data.get('terms', []))


def extract_and_save_domain_terms(
    data_path: str,
    output_path: str,
    p_value: float = 0.05,
) -> Set[str]:
    """
    training.jsonl'dan chi2 analiziyle domain terimlerini çıkarır,
    output_path'e JSON olarak kaydeder ve term set'ini döndürür.

    build_domain_terms.main() ve classifier.train_and_save() tarafından
    ortak kullanılır; domain_terms.json yokken otomatik oluşturmak için
    import edilebilir.
    """
    data_path = Path(data_path)
    output_path = Path(output_path)

    queries: list[str] = []
    labels: list[int] = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            queries.append(normalize_turkish(item['query']))
            labels.append(1 if item['label'] == 'complex' else 0)

    vectorizer = CountVectorizer(min_df=3, token_pattern=r'[a-zçğıöşü0-9]{3,}')
    X = vectorizer.fit_transform(queries)
    feature_names = vectorizer.get_feature_names_out()

    chi2_scores, p_values = chi2(X, labels)

    terms_info = [
        {'term': term, 'chi2': chi2_scores[i], 'p_value': p_values[i]}
        for i, term in enumerate(feature_names)
        if p_values[i] < p_value and term not in GENERIC_FILTER
    ]
    terms_info.sort(key=lambda x: x['chi2'], reverse=True)
    extracted_terms = [t['term'] for t in terms_info]

    output_data = {
        "version": "auto_v1",
        "extraction_method": f"chi2_p{str(p_value).replace('.', '')}",
        "source_data": str(data_path),
        "n_source_queries": len(queries),
        "n_terms": len(extracted_terms),
        "terms": extracted_terms,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    return set(extracted_terms)

def main():
    parser = argparse.ArgumentParser(description="Extract domain-specific terms from training data.")
    parser.add_argument('--data', type=str, default='data/decomposer_data/training.jsonl',
                        help='Training data path')
    parser.add_argument('--output', type=str, default='data/decomposer_data/domain_terms.json',
                        help='Output JSON path')
    parser.add_argument('--p-value', type=float, default=0.05,
                        help='Chi2 significance threshold')

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file {data_path} does not exist.")
        return

    extracted_terms = extract_and_save_domain_terms(
        data_path=args.data,
        output_path=args.output,
        p_value=args.p_value,
    )

    output_path = Path(args.output)
    with open(output_path, 'r', encoding='utf-8') as f:
        saved = json.load(f)

    n_queries = saved['n_source_queries']
    n_terms = saved['n_terms']

    print(f"Loaded {n_queries} queries from {args.data}")
    print(f"\nExtracted {n_terms} domain terms (p < {args.p_value})")
    print("-" * 60)
    print(f"{'Term':<20} | {'Chi2 Score':<15} | {'p-value':<15}")
    print("-" * 60)

    for term in saved['terms']:
        print(f"{term:<20}")
    print("-" * 60)
    print(f"Saved extracted terms to {output_path}")


if __name__ == '__main__':
    main()
