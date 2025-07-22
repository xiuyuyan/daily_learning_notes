import sqlite3
def query_by_product_name(product_name):
    """
    根据传入的产品名在数据库中搜索相关信息
    :param product_name:
    :return:
    """
    conn = sqlite3.connect('./data/SportsEquipment.db')
    cursor = conn.cursor()

    # 查询
    cursor.execute("SELECT * FROM products WHERE product_name LIKE ?", ('%' + product_name + '%',))

    rows = cursor.fetchall()

    conn.close()
    return rows


def read_store_promotions(product_name):
    # 指定优惠政策文档的文件路径
    file_path = './data/store_promotions.txt'

    try:
        # 打开文件并按行读取内容
        with open(file_path, 'r', encoding='utf-8') as file:
            promotions_content = file.readlines()

        # 搜索包含产品名称的行
        filtered_content = [line for line in promotions_content if product_name in line]

        # 返回匹配的行，如果没有找到，返回一个默认消息
        if filtered_content:
            return ''.join(filtered_content)
        else:
            return "没有找到关于该产品的优惠政策。"
    except FileNotFoundError:
        # 文件不存在的错误处理
        return "优惠政策文档未找到，请检查文件路径是否正确。"
    except Exception as e:
        # 其他潜在错误的处理
        return f"读取优惠政策文档时发生错误: {str(e)}"

def calculate(what):
    return eval(what)