from distribution import Distribution
import numpy as np

class ChineseRestaurantProcess(Distribution):

    def sample_one(self):
        N = self.get_parameter("N")

        tables = []
        table_counts = []
        n_tables = 0

        for i in xrange(N):
            probs = self._get_probabilities(table_counts)
            table = np.random.choice(np.arange(n_tables + 1), p=probs)
            if table == n_tables:
                tables.append({i})
                n_tables += 1
                table_counts.append(1)
            else:
                tables[table].add(i)
                table_counts[table] += 1

        return tables

    def _get_probabilities(self, table_counts):
        alpha = self.get_parameter("alpha")
        n_tables = len(table_counts)

        probs = np.zeros(n_tables + 1)
        probs[:-1] = table_counts

        probs += alpha
        return probs / float(sum(probs))

    def parameters(self):
        return {"alpha", "N"}


if __name__ == "__main__":
    crp = ChineseRestaurantProcess({
        'alpha': 1,
        'N': 100
    })
    print crp.sample(1)
