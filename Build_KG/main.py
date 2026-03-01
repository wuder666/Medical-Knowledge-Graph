import json
from py2neo import Graph
from tqdm import tqdm


def print_medical_inf(data_path):
    i = 0
    with open(data_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        print(f'lines的长度--》{len(lines)}')
        for line in lines:
            data = json.loads(line)
            print(json.dumps(data, indent=4, sort_keys=True, separators=(', ', ': '), ensure_ascii=False))
            i += 1
            print('*' * 80)
            if i == 5:
                break


class MedicalExtractor(object):
    def __init__(self):
        super(MedicalExtractor, self).__init__()
        # 1. 改用Bolt协议，删除database参数，不单独执行USE neo4j
        self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "123456"))

        # 共4类节点
        self.drugs = []  # 药品
        self.foods = []  # 食物
        self.diseases = []  # 疾病
        self.symptoms = []  # 症状

        # 构建节点实体关系
        self.rels_noteat = []  # 疾病－忌吃食物关系
        self.rels_doeat = []  # 疾病－宜吃食物关系
        self.rels_recommanddrug = []  # 疾病－热门药品关系
        self.rels_symptom = []  # 疾病症状关系

    def extract_triples(self, data_path):
        print(f'开始进行spo三元组的抽取')
        # 读取文件数据
        with open(data_path, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in tqdm(lines):
                data = json.loads(line)
                # 根据name获得疾病
                disease = data["name"]
                self.diseases.append(disease)

                # 处理症状
                if 'symptom' in data:
                    self.symptoms += data["symptom"]
                    for symptom in data["symptom"]:
                        self.rels_symptom.append([disease, 'has_symptom', symptom])

                # 并发症
                if "acompany" in data:
                    for acompany in data["acompany"]:
                        self.diseases.append(acompany)

                # 药品
                if 'recommand_drug' in data:
                    self.drugs += data["recommand_drug"]
                    for drug in data["recommand_drug"]:
                        self.rels_recommanddrug.append([disease, 'recommand_drug', drug])

                # 忌吃食物
                if 'not_eat' in data:
                    self.foods += data["not_eat"]
                    for _not in data["not_eat"]:
                        self.rels_noteat.append([disease, 'not_eat', _not])

                # 宜吃食物
                if 'do_eat' in data:
                    self.foods += data["do_eat"]
                    for _do in data["do_eat"]:
                        self.rels_doeat.append([disease, 'do_eat', _do])

                # 药物详情
                if 'drug_detail' in data:
                    for drug_d in data["drug_detail"]:
                        word_list = drug_d.split('(')
                        if len(word_list) == 2:
                            p, d = word_list
                            drug = d.rstrip(')')
                            self.drugs.append(drug)
                        else:
                            drug = word_list[0]
                            self.drugs.append(drug)

    def write_nodes(self, entities, entity_type):
        print(f"写入{entity_type}实体")
        # 去重 + 清理特殊字符（避免Cypher语法错误）
        clean_entities = set(
            [node.replace("'", "").replace('"', '').replace('(', '').replace(')', '') for node in entities])
        for node in tqdm(clean_entities):
            # 2. 将USE neo4j与MERGE合并执行（解决单独执行报错）
            sql = '''USE neo4j MERGE (n:{label_name}{{name:"{node_name}"}}) RETURN n'''.format(
                label_name=entity_type, node_name=node
            )
            try:
                self.graph.run(sql)
            except Exception as e:
                print(f"节点写入失败：{node}，错误：{e}")
                print(f"错误SQL：{sql}")

    def create_entities(self):
        # 必须先创建节点再创建关系
        self.write_nodes(self.drugs, '药品')
        self.write_nodes(self.symptoms, '症状')
        self.write_nodes(self.foods, '食物')
        self.write_nodes(self.diseases, '疾病')

    def write_relations(self, triples, head_type, tail_type):
        if not triples:
            print(f"警告：{head_type}-{tail_type}关系为空，跳过导入")
            return
        print(f"导入'{triples[0][1]}'关系类型的数据")
        for sub, relation, obj in tqdm(triples):
            # 清理实体名称特殊字符
            clean_sub = sub.replace("'", "").replace('"', '').replace('(', '').replace(')', '')
            clean_obj = obj.replace("'", "").replace('"', '').replace('(', '').replace(')', '')
            # 2. 将USE neo4j与MATCH/MERGE合并执行，添加RETURN避免语法错误
            sql = """USE neo4j MATCH (p:{head_type}),(q:{tail_type}) 
                     WHERE p.name='{sub_name}' AND q.name='{obj_name}'
                     MERGE (p) - [r:{relation}]-> (q)
                     RETURN r""".format(
                head_type=head_type, tail_type=tail_type,
                sub_name=clean_sub, obj_name=clean_obj, relation=relation
            )
            try:
                self.graph.run(sql)
            except Exception as e:
                print(f"关系写入失败：{clean_sub} -[{relation}]-> {clean_obj}，错误：{e}")
                print(f"错误SQL：{sql}")

    def create_spo(self):
        self.write_relations(self.rels_noteat, "疾病", '食物')
        self.write_relations(self.rels_doeat, "疾病", '食物')
        self.write_relations(self.rels_recommanddrug, "疾病", '药品')
        self.write_relations(self.rels_symptom, "疾病", '症状')


if __name__ == '__main__':
    # 初始化提取器
    kg = MedicalExtractor()

    # 可选：清空旧数据（首次运行建议执行，注意：会删除所有数据）
    # kg.graph.run("MATCH (n) DETACH DELETE n")

    # 1. 抽取三元组
    kg.extract_triples(data_path='medical.json')

    # 2. 创建节点（核心：必须先创建节点！）
    kg.create_entities()

    # 3. 创建关系
    kg.create_spo()

    # 验证数据写入结果
    node_count = kg.graph.run("MATCH (n) RETURN count(n) AS cnt").data()[0]['cnt']
    rel_count = kg.graph.run("MATCH ()-[r]->() RETURN count(r) AS cnt").data()[0]['cnt']
    print(f"\n数据导入完成！")
    print(f"当前数据库节点总数：{node_count}")
    print(f"当前数据库关系总数：{rel_count}")