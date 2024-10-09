from manim import *
from manim.utils.color import WHITE

color_list = [RED, GREEN, BLUE, YELLOW, PURPLE, ORANGE, PINK]

class SimpleArrow(Mobject):
    def __init__(self, start=ORIGIN, end=RIGHT, start_buff=0.25, end_buff=0.25, dot_scale=1, stroke_width=6, tip_length=0.4, **kwargs):
        super().__init__(**kwargs)
        
        # Ensure both start and end are 3D vectors
        start = self.ensure_3d(start)
        end = self.ensure_3d(end)

        # Calculate the direction of the arrow
        direction = end - start
        distance = np.linalg.norm(direction)
        unit_direction = direction / distance

        # Adjust start and end points with the given buffer (padding)
        start_adjusted = start + start_buff * unit_direction
        end_adjusted = end - end_buff * unit_direction

        # Create a dot at the original start point
        self.dot = Dot(start_adjusted).scale(dot_scale)
        
        # Create the arrow with the adjusted start and end positions
        self.arrow = Arrow(start_adjusted, end_adjusted, buff=0, stroke_width=stroke_width, tip_length=tip_length)  # buff=0 to avoid additional buffer from Arrow

        # Add both dot and arrow to the Mobject
        self.add(self.dot, self.arrow)

    def ensure_3d(self, point):
        """Converts 2D points to 3D by adding z=0 if needed"""
        if len(point) == 2:
            point = np.array([*point, 0])
        return point
    
    def create(self):
        # Use ShowCreation for the arrow and dot individually
        return AnimationGroup(Create(self.dot), GrowArrow(self.arrow), lag_ratio=0.5)

class ComplexCircle(Mobject):
    def __init__(self, radius=1, text="", color=WHITE, **kwargs):
        super().__init__(**kwargs)

        # Create the transfer function circle and store it as an instance attribute
        self.circle = Circle(radius=radius, color=color)
        self.circle.move_to(ORIGIN)  # Keep the circle at the origin

        # Create the inner text (inside the circle) and store it as an instance attribute
        self.text = MathTex(text)
        self.text.scale(radius * 1.5)  # Scale first, then position to avoid any movement issues
        self.text.move_to(self.circle.get_center())  # Center it within the circle

        # Add the circle and text to the Mobject
        self.add(self.circle, self.text)

    def get_radius(self):
        """Returns the radius of the circle."""
        return self.circle.get_radius()

    def get_text(self):
        """Returns the text Mobject."""
        return self.text[0]
    
    def create(self):
        """Returns an animation that creates the circle and writes the text."""
        return AnimationGroup(Create(self.circle), Write(self.text), lag_ratio=0.5)

class ComplexSquare(Mobject):
    def __init__(self, side_length=1, text="", color=WHITE, **kwargs):
        super().__init__(**kwargs)

        # Create the transfer function square and store it as an instance attribute
        self.square = Square(side_length=side_length, color=color)
        self.square.move_to(ORIGIN)  # Keep the square at the origin

        # Create the inner text (inside the square) and store it as an instance attribute
        self.text = MathTex(text)
        self.text.scale(side_length * 1.5)  # Scale first, then position to avoid any movement issues
        self.text.move_to(self.square.get_center())  # Center it within the circle

        # Add the square and text to the Mobject
        self.add(self.square, self.text)

    def get_side_length(self):
        """Returns the side length of the square."""
        return self.square.get_side_length()

    def get_text(self):
        """Returns the text Mobject."""
        return self.text[0]
    
    def create(self):
        """Returns an animation that creates the square and writes the text."""
        return AnimationGroup(Create(self.square), Write(self.text), lag_ratio=0.5)


class SingleNeuron(Scene):
    def construct(self):
        def percentage_to_y_position(percentage):
            """
            Converts a percentage (0 to 100) of the screen height to the corresponding Y position.
            100% is the top of the screen, 0% is the bottom.
            """
            return (percentage / 100 - 0.5) * config.frame_height
    
        # Weight list
        weights = []

        # Input list
        inputs = []

        # Input to Weight list
        input_to_weight_list = []

        # Weight to Neuron list
        weight_to_neuron_list = []

        # Create the activation
        activation = ComplexSquare(
            side_length=1.5, color=BLUE,
            text=f"\\varphi",
            )
        activation.move_to(RIGHT * 4.5)

        # Create the neuron
        neuron = ComplexCircle(
            radius=0.75, color=BLUE,
            text=r"\sum_{}^{}",
            )
        neuron.move_to(RIGHT * 1)

        # Create the weight
        weight = ComplexCircle(
            radius=0.5, color=WHITE,
            text=f"\\omega_{{{1}j}}",
            )
        weight.move_to(LEFT * 2.5)
        weight.get_text()[1].set_color(color_list[0])  # Color the index in red

        # Create the input
        input = MathTex(f"x_1")
        input.scale(0.8)
        input[0][1].set_color(color_list[0])
        input.move_to(LEFT * 6)

        # Create the threshold
        threshold = MathTex(f"\\theta_{{j}}")
        threshold.scale(0.8)
        threshold.move_to(activation.get_center() + DOWN * (activation.get_side_length() / 2 + 1.3))

        # Create the output
        output = MathTex(f"y_j")
        output.scale(0.8)
        output.move_to(activation.get_center() + RIGHT * (activation.get_side_length() / 2 + 1))

        # Create arrow input to weight
        input_to_weight = SimpleArrow(input.get_right(), weight.get_left())

        # Create arrow weight to neuron
        weight_to_neuron = SimpleArrow(weight.get_right(), neuron.get_left())

        # Create arrow neuron to activation
        neuron_to_activation = SimpleArrow(neuron.get_right(), activation.get_left())

        # Create arrow threshold to activation
        threshold_to_activation = SimpleArrow(threshold.get_top(), activation.get_bottom())

        # Create arrow activation to output
        activation_to_output = SimpleArrow(activation.get_right(), output.get_left(), stroke_width=6)

        # Create text for the input and weights
        input_text = MathTex("input")
        input_text.scale(neuron.get_radius())  # Scale first, then move
        input_text.move_to(input.get_center() + DOWN * (0.8))
        
        weight_text = MathTex("weight")
        weight_text.scale(neuron.get_radius())  # Scale first, then move
        weight_text.move_to(weight.get_center() + DOWN * (weight.get_radius() + 0.8))

        # Create the outer text (below the circle)
        neuron_text = MathTex("transfer function")
        neuron_text.scale(neuron.get_radius())  # Scale first, then move
        neuron_text.move_to(neuron.get_center() + DOWN * (neuron.get_radius() + 0.8))  # Place it below the circle

        # Create the outer text (below the square)
        activation_text = MathTex("activation function")
        activation_text.scale(activation.get_side_length() / 2)  # Scale first, then move
        activation_text.move_to(activation.get_center() + UP * (activation.get_side_length() / 2 + 0.8))  # Place it below the circle

        # Create the threshold text
        threshold_text = MathTex("threshold")
        threshold_text.scale(neuron.get_radius())
        threshold_text.move_to(threshold.get_center() + DOWN * 0.8)

        # Create the output text
        output_text = MathTex("output")
        output_text.scale(neuron.get_radius())
        output_text.move_to(output.get_center() + DOWN * 0.8)

        self.play(
            Create(input_text),
            Create(weight_text),
            neuron.create(),
            Create(neuron_text),
            weight.create(),
            Create(input),
            input_to_weight.create(),
            weight_to_neuron.create(),
            activation.create(),
            neuron_to_activation.create(),
            Create(activation_text),
            Create(threshold),
            threshold_to_activation.create(),
            Create(output),
            Create(output_text),
            Create(threshold_text),
            activation_to_output.create()
        )

        # Display the circle
        #self.play(Create(weight))
        # Wait for a moment
        self.wait(5)

        # Generate N values for the input
        N = 5

        # Define the minimum and maximum Y positions in screen space
        min_y_position = percentage_to_y_position(15)  # 20% from the bottom
        max_y_position = percentage_to_y_position(75)   # 80% from the top

        # Move First element
        # Move element
        new_position, scale_factor = SingleNeuron.calculate_transform(0, N, min_y_position, max_y_position)

        # Take into account that position is lower that the original
        distance = weight.get_center() - LEFT * 0.5

        # Create new cells
        new_input = input.copy().shift(new_position - distance).scale(scale_factor * 3 * 1.25)
        new_weight = weight.copy().shift(new_position - distance).scale(scale_factor * 4)

        # Create new arrows
        new_input_to_weight = SimpleArrow(new_input.get_right(), new_weight.get_left(), dot_scale=0.5, stroke_width=3, tip_length=0.10)
        new_weight_to_neuron = SimpleArrow(new_weight.get_right(), neuron.get_left(), dot_scale=0.5, stroke_width=3, tip_length=0.10)

        # Get the X-coordinate from the input object, but preserve the Y-coordinate of new_input_text
        new_input_text = input_text.copy().scale(0.8).move_to([new_input.get_center()[0], percentage_to_y_position(90), 0])

        # Get the X-coordinate from the weight object, but preserve the Y-coordinate of new_weight_text
        new_weight_text = weight_text.copy().scale(0.8).move_to([new_weight.get_center()[0], percentage_to_y_position(90), 0])

        # Reposition and scale weight, input, and input_to_weight arrow
        self.play(
            Transform(weight, new_weight),
            Transform(input, new_input),
            Transform(input_to_weight, new_input_to_weight),
            Transform(weight_to_neuron, new_weight_to_neuron),
            Transform(input_text, new_input_text),
            Transform(weight_text, new_weight_text)
        )

        # Add to the lists
        weights.append(new_weight)
        inputs.append(new_input)
        input_to_weight_list.append(new_input_to_weight)
        weight_to_neuron_list.append(new_weight_to_neuron)  

        for i in range(1, N):
             # Create the weight
            weight = ComplexCircle(
                radius=0.5, color=WHITE,
                text=f"\\omega_{{{i + 1}j}}",
                )
            weight.move_to(LEFT * 1 + DOWN * (config.frame_height/2 - weight.get_radius()))
            weight.get_text()[1].set_color(color_list[i])  # Color the index in red

            # Create the input
            input = MathTex(f"x_{{{i + 1}}}")
            input.scale(0.8)
            input[0][1:].set_color(color_list[i])
            input.move_to(LEFT * 4.5 + DOWN * (config.frame_height/2 - weight.get_radius()))

            # Create arrow input to weight
            input_to_weight = SimpleArrow(input.get_right(), weight.get_left())

            # Create arrow weight to neuron
            weight_to_neuron = SimpleArrow(weight.get_right(), neuron.get_left())

            # Display the weight, input, and arrows
            self.play(
                weight.create().set_run_time(0.5),
                Create(input, run_time=0.5),
                input_to_weight.create().set_run_time(0.5),
                weight_to_neuron.create().set_run_time(0.5)
            )

            # Move element
            new_position, scale_factor = SingleNeuron.calculate_transform(i, N, min_y_position, max_y_position)

            # Take into account that position is lower that the original
            distance = weight.get_center() - LEFT * 0.5

            # Create new cells
            new_input = input.copy().shift(new_position - distance).scale(scale_factor * 3 * 1.25)
            new_weight = weight.copy().shift(new_position - distance).scale(scale_factor * 4)

            # Create new arrows
            new_input_to_weight = SimpleArrow(new_input.get_right(), new_weight.get_left(), dot_scale=0.5, stroke_width=3, tip_length=0.10)
            new_weight_to_neuron = SimpleArrow(new_weight.get_right(), neuron.get_left(), dot_scale=0.5, stroke_width=3, tip_length=0.10)

            # Reposition and scale weight, input, and input_to_weight arrow
            self.play(
                Transform(weight, new_weight),
                Transform(input, new_input),
                Transform(input_to_weight, new_input_to_weight),
                Transform(weight_to_neuron, new_weight_to_neuron),
            )

            # Add to the lists
            weights.append(new_weight)
            inputs.append(new_input)
            input_to_weight_list.append(new_input_to_weight)
            weight_to_neuron_list.append(new_weight_to_neuron)       

        self.wait(2)

    def calculate_transform(i, N, min_y_position, max_y_position):
        """
        Calculate the transformation for the i-th element out of N elements.
        """
        # Calculate new position (evenly distributed vertically)
        normalized_position = i / (N - 1)  # Normalized between 0 and 1
        new_y_position = interpolate(max_y_position, min_y_position, normalized_position)  # Constrain between min and max

        # Move the element to the exact new position
        new_position = UP * new_y_position + 2 * LEFT  # Adjust vertical position and shift left

        # Calculate scale factor
        scale_factor = 1 / N  # Scale inversely proportional to number of elements

        return new_position, scale_factor         
        