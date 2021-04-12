"""
- File: graph_wrapper.py
- Author: Jason A. F. Mitchell
- Summary: This module contains a wrapper class for creating graphs using 
           matplotlib. 
"""

import matplotlib.pyplot as plt

STEP_SIZE = 0.001


class Plot:
    """
    This class acts as a wrapper for the matplotlib pylot API.

    Attributes:
        delayTime(float): amount of seconds to interupt program when
                          displaying a graph
        
    Methods:
        addBarGraph(list,list) -> void
        addFunction(func,float,float,str,float) -> void
        addLegendLabel(str, int, str) -> void
        addLine(float, float, float, float, str) -> void
        addPoint(float, float, str, str) -> void
        addText(float, float, str) -> void
        clear() -> void
        display() -> void
        freeze() -> void
        label(str, str, str) -> void
        pointplot(list, list, str) -> void
        removeBarGraph(int) -> void
        removeLine(int) -> void
        removePoint(int) -> void
        removeText(int) -> void
        save(str) -> void
        setAxis(bool) -> void
        showLegend() -> void
    """
    def __init__(self):
        """
        Constructor for class.
        """        
        self._bars = []
        self._lines = []
        self._points = []
        self._text = []
        self.delayTime = 0.1

    def __del__(self):
        """
        Destructor for class.
        """        
        self.clear()

    def addBarGraph(self, xVals, yVals):
        """
        Plots a bar graph on the graph.

        Args:
            xVals (list): list of x values
            yVals (list): list of y values
        """        
        self._bars.append(plt.bar(xVals, yVals))

    def addFunction(self,
                    func,
                    xMin=0.0,
                    xMax=0.0,
                    color="red",
                    stepSize=STEP_SIZE):
        """
        Plots a linear function on the graph.

        Args:
            func (function): a function that returns a float Y when given a float X
            xMin (float, optional): Lower bound of x for plot. Defaults to 0.
            xMax (float, optional): Upper bound of x for plot. Defaults to 0.
            color (str, optional): Color of the plotted function line. Defaults to "red".
            stepSize (float, optional): Step size of x. Defaults to STEP_SIZE.
        """        
        newLine = [[], []]
        index = 0
        x = xMin
        while x < xMax:
            newLine[0].append(x)
            newLine[1].append(func(x))
            x += stepSize
            index += 1
        self._lines.append(plt.plot(newLine[0], newLine[1], color=color))

    def addLegendLabel(self, objectType="line", index=-1, label=""):
        """
        Adds a legend label for a line or point on the graph.

        Args:
            objectType (str, optional): Type of item to label, either "line" 
                                        or "point". Defaults to "line".
            index (int, optional): Index of item in item collection. Defaults 
                                   to the most recent item.
            label (str, optional): Label to give line or point. Defaults to "".
        """        
        objectArray = []
        if objectType == "point":
            objectArray = self._points
        else:
            objectArray = self._lines
        if index == -1:
            index = len(objectArray) - 1
        objectArray[index][0].set_label(label)

    def addLine(self, x1=0, y1=0, x2=0, y2=0, color="red"):
        """
        Adds a line segment to the graph.

        Args:
            x1 (int, optional): Initial x value. Defaults to 0.
            y1 (int, optional): Initial y value. Defaults to 0.
            x2 (int, optional): Terminating x value. Defaults to 0.
            y2 (int, optional): Terminating y value. Defaults to 0.
            color (str, optional): Color of line. Defaults to "red".
        """
        self._lines.append(plt.plot([x1, x2], [y1, y2], color=color))

    def addPoint(self, xVal=0, yVal=0, style="o", color="blue"):
        """
        Add a point to a graph
        Args:
            xVal (int, optional): [description]. Defaults to 0.
            yVal (int, optional): [description]. Defaults to 0.
            style (str, optional): Style of point (options: https://matplotlib.org/stable/api/markers_api.html). 
                                   Defaults to "o".
            color (str, optional): [description]. Defaults to "blue".
        """
        self._points.append(
            plt.plot(xVal, yVal, marker=style, color=color)
        )

    def addText(self, xVal=0, yVal=0, message=""):
        """
        Add text go graph.

        Args:
            xVal (int, optional): X val of text's position. Defaults to 0.
            yVal (int, optional): Y val of text's position. Defaults to 0.
            message (str, optional): Text to display. Defaults to "".
        """
        self._text.append(plt.text(xVal, yVal, message))

    def clear(self):
        """
        Clears graph of all markings.
        """
        while len(self._lines) > 0:
            self.removeLine()
        while len(self._points) > 0:
            self.removePoint()
        while len(self._text) > 0:
            self.removeText()

    def display(self):
        """
        Displays the current version of the graph for the time indicated in delayTime.
        """        
        plt.pause(self.delayTime)

    def freeze(self):
        """
        Displays the current version of the graph until user closes the popup window.
        """
        plt.show()

    def label(self, title="", xLabel="", yLabel=""):
        """
        Edits title, x-axis label, and y-axis label for the graph.

        Args:
            title (str, optional): Title of graph displayed at top. Defaults to "".
            xLabel (str, optional): X-axis label of graph. Defaults to "".
            yLabel (str, optional): Y-axis label of graph. Defaults to "".
        """
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)

    def pointplot(self, xArray, yArray, color="red"):
        """
        Adds multiple points to a graph with a given color.
        Args:
            xArray (list): X positions of points.
            yArray (list): Y positions of points.
            color (str, optional): Color of points. Defaults to "red".
        """
        self._lines.append(plt.plot(xArray, yArray, color=color))

    def removeBarGraph(self, index=-1):
        """
        Removes a bar graph from graph.

        Args:
            index (int, optional): Index of bar graph to remove. Defaults to most recent addition.
        """
        self._bars.pop(index).remove()

    def removeLine(self, index=-1):
        """
        Removes a line from graph.

        Args:
            index (int, optional): Index of line to remove. Defaults to most recent addition.
        """
        self._lines.pop(index).pop().remove()

    def removePoint(self, index=-1):
        """
        Removes a point from graph.

        Args:
            index (int, optional): Index of point to remove. Defaults to most recent addition.
        """
        self._points.pop(index).pop().remove()

    def removeText(self, index=-1):
        """
        Removes text from graph.

        Args:
            index (int, optional): Index of text to remove. Defaults to most recent addition.
        """
        self._text.pop(index).remove()

    def save(self, name="plot.jpg"):
        """
        Saves graph to a directory called "images/" located in the same directory
        as this file.

        Args:
            name (str, optional): Name of file. Extentension of file determines 
                                  type generated. Defaults to "plot.jpg".
        """
        plt.savefig("images/" + name)

    def setAxis(self, value=True):
        """
        Sets whether or not a graph shows an axis. Defaults to true.

        Args:
            value (bool, optional): Sets whether or not graph shows an axis. Defaults to True.
        """
        plt.axis(value)

    def showLegend(self):
        """
        Generates legend on graph. Must be called if user wants and added legend labels to show.
        """
        plt.legend()
