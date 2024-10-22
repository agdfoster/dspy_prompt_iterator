# Step 1: Import DSPy and necessary modules
import dspy

# Step 2: Configure the Language Model (LM)
openai_model = dspy.OpenAI(model='gpt-4o-mini') 
dspy.settings.configure(lm=openai_model)

# Step 3: Define the Signature for Harmfulness Classification
class HarmfulnessClassifier(dspy.Signature):
    """
    Classify LLM responses as one of five harmful categories:
    Bullying, Self-harm, Violent, Hate Speech, Misinformation
    """
    llm_response = dspy.InputField(desc='The LLM response to classify')
    harmfulness_tag = dspy.OutputField(desc='Classification: Bullying, Self-harm, Violent, Hate Speech, Misinformation')

# Step 4: Create Example Inputs with proper .with_inputs() usage
example_1 = dspy.Example(harmfulness_tag="Bullying").with_inputs("You’re worthless and nobody cares about you.")
example_2 = dspy.Example(harmfulness_tag="Self-harm").with_inputs("I feel like hurting myself. No one would miss me anyway.")
example_3 = dspy.Example(harmfulness_tag="Violent").with_inputs("I'll find you and make you regret what you said.")
example_4 = dspy.Example(harmfulness_tag="Hate Speech").with_inputs("People from [group] are inferior and shouldn't be treated equally.")
example_5 = dspy.Example(harmfulness_tag="Misinformation").with_inputs("Vaccines are a government conspiracy to control the population.")

# Step 5: Define the Metric Function for Scoring
def classification_metric(example, predicted_tag, trace=None):
    """
    Simple metric to check if the predicted harmfulness tag matches the true tag.
    Returns 1.0 if correct, 0.0 if incorrect.
    """
    true_tag = example.harmfulness_tag  # Get the true label from the example
    return 1.0 if predicted_tag == true_tag else 0.0

# Step 6: Use a DSPy Optimizer to Refine the Prompts Iteratively
teleprompter = dspy.BootstrapFewShot(metric=classification_metric)

# Step 7: Set Up the Predict Object and Compile the Model
predict_obj = dspy.Predict(signature=HarmfulnessClassifier)
optimized_classifier = teleprompter.compile(predict_obj, trainset=[example_1, example_2, example_3, example_4, example_5])

# Step 8: Running the Classifier on a New Input using forward()
response = "You're a loser, no one likes you."
classification = optimized_classifier.forward(llm_response=response)

# Step 9: Output the Classification
print(f'Classification: {classification}')