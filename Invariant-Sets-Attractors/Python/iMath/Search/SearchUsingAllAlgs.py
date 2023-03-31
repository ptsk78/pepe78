import threading

from iMath.Search.BaseSearch import BaseSearch
from iMath.Search.SpreadBoxSearch.SpreadBoxSearch import SpreadBoxSearch
from iMath.Search.KDTree import KDTree


class TMP:
    vp1 = None
    vp2 = None
    ret = None


class SearchUsingAllAlgs:
    bs_wins = 0
    sbs_wins = 0
    kdt_wins = 0

    @staticmethod
    def get_results_bs(obj):
        bs = BaseSearch(obj.vp2)
        obj.ret = bs.get_closest_points(obj.vp1)

    @staticmethod
    def get_results_sbs(obj):
        sbs = SpreadBoxSearch(obj.vp2, obj.vp1)
        obj.ret = sbs.get_closest_points(obj.vp1)

    @staticmethod
    def get_results_kdt(obj):
        sbs = KDTree(obj.vp2)
        obj.ret = sbs.get_closest_points(obj.vp1)

    @staticmethod
    def get_best_indexes(vp1, vp2):
        obj1 = TMP()
        obj1.vp1 = vp1
        obj1.vp2 = vp2
        th1 = threading.Thread(target=SearchUsingAllAlgs.get_results_bs, args=(obj1,))
        obj2 = TMP()
        obj2.vp1 = vp1
        obj2.vp2 = vp2
        th2 = threading.Thread(target=SearchUsingAllAlgs.get_results_sbs, args=(obj2,))
        obj3 = TMP()
        obj3.vp1 = vp1
        obj3.vp2 = vp2
        th3 = threading.Thread(target=SearchUsingAllAlgs.get_results_kdt, args=(obj3,))

        th1.start()
        th2.start()
        th3.start()

        while True:
            if not th1.is_alive():
                which = 0
                break
            if not th2.is_alive():
                which = 1
                break
            if not th3.is_alive():
                which = 2
                break

        if which == 0:
            SearchUsingAllAlgs.bs_wins += 1
            ret = obj1.ret
        else:
            if which == 1:
                SearchUsingAllAlgs.sbs_wins += 1
                ret = obj2.ret
            else:
                SearchUsingAllAlgs.kdt_wins += 1
                ret = obj3.ret
        return ret
