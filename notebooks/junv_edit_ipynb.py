import json

notebook_path = 'notebooks/junv_dgdata_crawling.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        source_str = "".join(source)
        
        if "app.scrape_url(" in source_str:
            # We want to replace the relevant block
            old_code = """        try:
            result = app.scrape_url(
                url, 
                params={
                    'formats': ['extract'],
                    'extract': {
                        'schema': DaangnExtractSchema.model_json_schema(),
                        'prompt': "당근마켓 페이지의 검색결과 목록에서 '폴로' 상품들의 ID, 이름, 가격, 판매상태, 등록시간, 사진URL을 모두 추출해줘."
                    },
                    'waitFor': 3000
                }
            )
            
            if 'extract' in result and 'items' in result['extract']:
                items = result['extract']['items']"""

            new_code = """        try:
            result = app.scrape(
                url, 
                formats=[{
                    'type': 'json',
                    'schema': DaangnExtractSchema.model_json_schema(),
                    'prompt': "당근마켓 페이지의 검색결과 목록에서 '폴로' 상품들의 ID, 이름, 가격, 판매상태, 등록시간, 사진URL을 모두 추출해줘."
                }],
                wait_for=3000
            )
            
            if hasattr(result, 'json') and result.json and 'items' in result.json:
                items = result.json['items']"""
            
            if old_code in source_str:
                new_source_str = source_str.replace(old_code, new_code)
                # Split back into lines maintaining the newlines at the end of each string
                lines = []
                for i, line in enumerate(new_source_str.split('\n')):
                    if i < len(new_source_str.split('\n')) - 1:
                        lines.append(line + '\n')
                    else:
                        if line:
                            lines.append(line)
                            
                cell['source'] = lines
                print("Successfully updated the scraping logic in the notebook.")
            else:
                print("Code to replace not found exactly as expected.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)
