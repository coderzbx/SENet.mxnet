# -*-coding:utf-8-*-

from collections import namedtuple


Label = namedtuple(
    'Label', ['id', 'categoryId', 'label', 'name'])


main_assist_labels = {
    Label(0, 0, "0", u"主路"),
    Label(1, 1, "1", u"辅路"),
}