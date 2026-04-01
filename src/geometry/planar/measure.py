

from .line import Line2D
from .point import Point2D, Points2D


class Geometry2DMeasure:
    @staticmethod
    def measure_distance(
        input1: Line2D | Point2D | Points2D, 
        input2: Line2D | Point2D | Points2D
        ) -> float:
        """
        Measure the distance between two input shapes.

        Parameters
        ----------
        input1: Line2D | Point2D | Points2D
            The first input shape.
        input2: Line2D | Point2D | Points2D
            The second input shape.

        Returns
        -------
        float: The distance between the two input shapes.
        """
        return input1.shapely.distance(input2.shapely)

    # @staticmethod
    # def measure_intersection(input1: Polygon, input2: Polygon) -> float:
    #     return input1.intersection(input2).area

    # @staticmethod
    # def measure_union(input1: Polygon, input2: Polygon) -> float:
    #     return input1.union(input2).area
