
class IBlock(): # blocs les uns au dessus des autres, fait pour que l'ia apprenne à sauter dessus pour passer des obstacles par exemple
    def __init__(self,id,height=1):
        self.id=id
        self.height=height
