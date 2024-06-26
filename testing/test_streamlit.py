import streamlit.testing.v1 as st_test
from streamlit.testing.v1 import AppTest

class MyAppTest(AppTest):
    def run_app(self):
        import app
        app.main() #TODO: main should recieve patient args

    def test_button_clicks(self):
        button_labels = [
            'button1', 'button2'
        ] #TODO: add
        for button in button_labels:
            try:
                with self.run_app_in_context() as test:
                    # Simulate button click
                    test.click(button)
                    # Check that no exceptions have occurred
                    assert True  # If no exception occurs, the test passes
            except Exception as e:
                # If an exception occurs, the test fails
                assert False, f"Exception occurred when clicking {button}: {e}"


    def test_slider_interactions(self):
        sliders = {
            # 'Slider 1': (0, 100, 25),  # (min, max, default)
            # 'Slider 2': (0, 50, 10),
            # 'Slider 3': (10, 20, 15)
        } #TODO: add
        for slider_label, (min_val, max_val, default_val) in sliders.items():
            try:
                with self.run_app_in_context() as test:
                    # Simulate slider interaction by setting it to the max value
                    test.slider(slider_label, value=max_val)
                    # Check that no exceptions have occurred
                    assert True  # If no exception occurs, the test passes
            except Exception as e:
                # If an exception occurs, the test fails
                assert False, f"Exception occurred when interacting with {slider_label}: {e}"
if __name__ == '__main__':
    MyAppTest().run()
