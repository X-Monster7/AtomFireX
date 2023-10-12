# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/9/19 20:41
@version: 1.0.0
"""
import requests
import json


def title2info(len_of_max_result: int, title: str, *args, **kwargs) -> dict:
    """
    将文献的标题（必须）、作者以及其它信息作为条件，进行查询。
    Args:
        len_of_max_result: 查询的最大结果数
        title (): necessary
        *args ():
        **kwargs ():

    Returns:
        return {'error': False, 'abstract': abstract, 'doi': doi, 'issue': issue,
                    'volume': volume, 'page': page }
    """
    params = {
        'query.title': title,
        # 'query.author': args.author
        # 'query.description': args.description
    }
    url = 'https://api.crossref.org/works?query.title={}'.format(params['query.title'])
    LEN_OF_MAX_RESULT = len_of_max_result

    # ========================================================

    # 创建一个Session对象，该会话不使用系统代理
    session = requests.Session()
    session.trust_env = False
    session.proxies = {}  # 清空代理配置

    try:
        response = session.get(url, timeout=5000)
        # 检查响应是否成功
        if response.status_code == 200:
            # 解析 JSON 数据
            queryed_data = json.loads(response.text)['message']['items'][:LEN_OF_MAX_RESULT]
            for data_ in queryed_data:
                if data_['title'][0] != params.get('query.title'):
                    continue
                abstract, doi, issue, volume, page = data_.get('abstract', ''), data_.get('DOI', ''), \
                    data_.get('issue', ''), data_.get('volume', ''), data_.get('page', '')
                print('doi: {}, abstract:{}'.format(doi, abstract))
                return {'error': False, 'abstract': abstract, 'doi': doi, 'issue': issue,
                        'volume': volume, 'page': page}
            print(f'{title} not found')
            return {'error': True}
            # 将数据写入本地文件
            # with open("demo_data.json", "w", encoding="utf-8") as f:
            #     json.dump(data, f, ensure_ascii=False)
        else:
            # 响应失败
            print(f"title2doi请求失败，状态码：{response.status_code}")
            return {'error': True}
    except BaseException:
        with open(f"./error/{title}_error", "w", encoding="utf-8") as f:
            f.write(f'{title} error')
            return {'error': True}



if __name__ == '__main__':
    print(title2info(10, 'Molecule optimization by explainable evolution'))
