"""
    instance_id,

    item_id,item_category_list,item_property_list,item_brand_id,item_city_id,
    item_price_level,item_sales_level,item_collected_level,item_pv_level,

    user_id,user_gender_id,user_age_level,user_occupation_id,user_star_level,

    context_id,context_timestamp,context_page_id,predict_category_property,

    shop_id,shop_review_num_level,shop_review_positive_rate,shop_star_level,
    shop_score_service,shop_score_delivery,shop_score_description,

    is_trade

    From the step1, we can see the user_id has the around num with instance_id
    So the item propery or category may count the cvr
    And the shop_id, item_id could be one-hot
"""
import time
import pandas as pd

tic = time.process_time()
SAMPLE_TRAIN_FILE = 'sample_train.csv'
FEATURE_TRAIN_FILE = 'feature_train.csv'
traindf = pd.read_csv(SAMPLE_TRAIN_FILE)

counter_cate_dict = {}  # all the counter for properties of items dict
counter_prop_dict = {}  # all the counter for properties of items dict
counter_pre_cate_dict = {}  # all the pre categories of items in dict
counter_pre_prop_dict = {}  # all the pre properties of items in dict


# context_timestamp time
# item_category_list,item_property_list
# predict_category_property
def dealitemcateprop(itemrow, debug=False):
    # instance_id
    instance_id = itemrow['instance_id']
    # context_timestamp 广告商品的展示时间，Long类型；取值是以秒为单位的Unix时间戳
    tm = time.localtime(itemrow['context_timestamp'])
    day = 0 if tm.tm_mday == 31 else tm.tm_mday
    hour = tm.tm_hour
    hour48 = tm.tm_hour * 2 + tm.tm_min // 30

    # "category_0;category_1;category_2"
    cate_list = itemrow['item_category_list'].split(";")

    # "property_0;property_1;property_2"
    prop_list = itemrow['item_property_list'].split(";")

    pre_cate_list = []
    pre_prop_list = []
    # "category_A:property_A_1,property_A_2,property_A_3;category_B:-1;category_C:property_C_1,property_C_2"
    for cps in itemrow['predict_category_property'].split(";"):
        pre_cate_list.append(cps.split(":")[0])
        tmplist = cps.split(":")[-1].split(",")
        if tmplist[0] != '-1':
            pre_prop_list += tmplist

    # the cross set
    same_cate = len(set(cate_list) ^ set(pre_cate_list))
    same_prop = len(set(prop_list) ^ set(pre_prop_list))

    if debug:
        print("item_category_list: \n", itemrow['item_category_list'])
        print("cate_list: \n", cate_list)
        print("item_property_list: \n", itemrow['item_property_list'])
        print("prop_list: \n", prop_list)
        print("predict_category_property: \n", itemrow['predict_category_property'])
        print("pre_cate_list: \n", pre_cate_list)
        print("pre_prop_list: \n", pre_prop_list)

    return instance_id, \
           cate_list, prop_list, pre_cate_list, pre_prop_list, \
           same_cate, same_prop, \
           day, hour, hour48


featdf = pd.DataFrame(list(traindf.apply(lambda x: dealitemcateprop(x), axis=1)),
                      columns=['instance_id',
                               'cate_list', 'prop_list', 'pre_cate_list', 'pre_prop_list',
                               'same_cate', 'same_prop',
                               'day', 'hour', 'hour48'])


def caldicts(featdf):
    global counter_cate_dict, counter_prop_dict, counter_pre_cate_dict, counter_pre_prop_dict
    for _, row in featdf.iterrows():
        for val in row['cate_list']:
            counter_cate_dict[val] = counter_cate_dict.get(val, 0) + 1
        for val in row['prop_list']:
            counter_prop_dict[val] = counter_prop_dict.get(val, 0) + 1
        for val in row['pre_cate_list']:
            counter_pre_cate_dict[val] = counter_pre_cate_dict.get(val, 0) + 1
        for val in row['pre_prop_list']:
            counter_pre_prop_dict[val] = counter_pre_prop_dict.get(val, 0) + 1


caldicts(featdf=featdf)

print("The SET: ")
print(len(counter_cate_dict))  # , counter_cate_dict)
print(len(counter_prop_dict))  # , counter_prop_dict)
print(len(counter_pre_cate_dict))  # , counter_pre_cate_dict)
print(len(counter_pre_prop_dict))  # , counter_pre_prop_dict)
# The SET:86 129888 1244 2221
print(featdf.head(1))

TOPK = 5
# 'cate_list', 'prop_list', 'pre_cate_list', 'pre_prop_list'
def topfeat(itemrow):
    # instance_id
    instance_id = itemrow['instance_id']

    def topclasses(clslist, clsctrdict, topk):
        retsort = sorted(clslist, key=lambda x: clsctrdict[x], reverse=True)
        retdict = {}
        for i in range(topk):
            retdict[i] = "_".join(retsort[:i + 1])
        return retdict

    # 'cate_list'
    clslist = itemrow['cate_list']
    clsctrdict = counter_cate_dict
    cate_topk_dict = topclasses(clslist=clslist, clsctrdict=clsctrdict, topk=TOPK)

    # 'prop_list'
    clslist = itemrow['prop_list']
    clsctrdict = counter_prop_dict
    prop_topk_dict = topclasses(clslist=clslist, clsctrdict=clsctrdict, topk=TOPK)

    # 'pre_cate_list'
    clslist = itemrow['pre_cate_list']
    clsctrdict = counter_pre_cate_dict
    pre_cate_topk_dict = topclasses(clslist=clslist, clsctrdict=clsctrdict, topk=TOPK)

    # 'pre_prop_list'
    clslist = itemrow['pre_prop_list']
    clsctrdict = counter_pre_prop_dict
    pre_prop_topk_dict = topclasses(clslist=clslist, clsctrdict=clsctrdict, topk=TOPK)

    return (instance_id,
            *cate_topk_dict.values(), *prop_topk_dict.values(),
            *pre_cate_topk_dict.values(), *pre_prop_topk_dict.values())


topfeatdf = pd.DataFrame(list(featdf.apply(lambda x: topfeat(x), axis=1)),
                         columns=['instance_id'] +
                                 ['cate_top' + str(i) for i in range(TOPK)] +
                                 ['prop_top' + str(i) for i in range(TOPK)] +
                                 ['pre_cate_top' + str(i) for i in range(TOPK)] +
                                 ['pre_prop_top' + str(i) for i in range(TOPK)])

data=pd.merge(traindf, featdf, on='instance_id', how='left')
data=pd.merge(data, topfeatdf, on='instance_id', how='left')
data.to_csv(FEATURE_TRAIN_FILE, index=False)
print("Last: ", time.process_time() - tic)
