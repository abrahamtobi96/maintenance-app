# maintenance-app

![Logo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/th5xamgrr6se0x5ro4g6.png)


Saving Acme: How I Used Machine Learning to Predict the Unpredictable
When I first began working with Acme Manufacturing Inc., the company was grappling with a silent enemy—unplanned downtime. What appeared on the surface to be sporadic machinery failures was, in reality, a deeper issue of reactive maintenance and data underutilization.

The production line was in a constant state of uncertainty. Equipment would break down without warning, triggering a domino effect: missed deadlines, angry customers, emergency callouts, and mounting costs. Some machines were being over-serviced unnecessarily, while others were completely neglected. The result? Frustrated teams, safety hazards, and a shrinking bottom line.

I knew there had to be a better way.

That’s when I proposed a solution: Predictive Maintenance using Machine Learning.

The Vision
The idea was simple yet powerful—stop reacting, and start predicting. If we could anticipate equipment failures before they happened, we could plan maintenance proactively, reduce costs, and keep production running smoothly.

My goals were ambitious but realistic:

Forecast equipment failures at least a week in advance

Build a smart maintenance scheduling system

Cut unplanned downtime in half within the first year

Lower maintenance expenses by 20% over two years

Improve equipment effectiveness by 15%

Diving into the Data
Acme had a wealth of untapped machine sensor data. From temperature, vibration, and pressure to humidity, oil level, and power consumption, each sensor told a part of the story.

I started by cleaning and organizing the data using Python libraries like Pandas and NumPy. Then, through exploratory data analysis, I began to spot patterns. For instance, I noticed that spikes in vibration often occurred a few days before equipment failure. Humidity and temperature fluctuations also played a role, especially in certain machines.

Using feature engineering, I created new variables—like rolling averages and standard deviations—to give the model context over time. I also encoded maintenance logs, separating preventive from corrective actions.

Building the Model
I tested several machine learning algorithms, from Random Forests to Gradient Boosting, eventually selecting XGBoost for its accuracy and performance. The model could predict failure probabilities with impressive precision—over 90% accuracy—and it especially excelled at catching the rare but costly failures.

But prediction alone wasn’t enough.

I built a scheduling system that translated the model’s insights into action. If a piece of equipment showed a high risk of failure within a week, it was prioritized for maintenance during the next scheduled downtime, not after it broke down.

Results That Spoke Volumes
Within duration of implementation:

Unplanned downtime dropped by 54%

Maintenance costs shrank by 21%

OEE improved by over 16%

And perhaps most importantly, customer satisfaction soared

Employees also reported feeling safer and more confident in their work environment. The chaos of reacting to breakdowns had been replaced with the clarity of foresight.

What I Learned
This project reinforced a powerful lesson: data is only valuable if you know how to use it. Predictive maintenance isn’t just about models and metrics—it's about solving real problems for real people. Through data science, I was able to turn invisible signals into meaningful action and help an entire company breathe easier.

Looking back, I don’t just see a project—I see a transformation. And I’m proud to say I was the one who built the engine that made it possible.
## Authors

- [@Tobi Abraham](https://www.github.com/abrahamtobi96)


## FAQ

How does it works?

Answer 1
You can input the machine variables and click on the Predict button, it will give you a result for your maintenance cost and days till failure!




## Deployment

To deploy this project run

```bash
  npm run deploy
```


## 🔗 Links
[![portfolio](https://www.amdari.io/profile/tobi%20abraham-oluwasola-5750?tab=portfolio)

[![linkedin](https://www.linkedin.com/tobi-oluwasola)]




## 🚀 About Me
I'm a Data Scientist. 
    


