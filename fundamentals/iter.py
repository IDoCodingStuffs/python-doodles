from datetime import timedelta

class DateRangeIterable:
    """An iterable that contains its own iterator object."""

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self._present_day = start_date

    def __iter__(self):
        return self

    def __next__(self):
        if self._present_day >= self.end_date:
            raise StopIteration()
        today = self._present_day
        self._present_day += timedelta(days=1)
        return today

from datetime import date
for day in DateRangeIterable(date(2018, 1, 1), date(2018, 1, 5)):
    print(day)


class DateRangeContainerIterable:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def __iter__(self):
        current_day = self.start_date
        while current_day < self.end_date:
            yield current_day
            current_day += timedelta(days=1)


r1 = DateRangeContainerIterable(date(2018, 1, 1), date(2018, 1, 5))
print(", ".join(map(str, r1)))
print(max(r1))

class DateRangeSequence:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self._range = self._create_range()

    def _create_range(self):
        days = []
        current_day = self.start_date
        while current_day < self.end_date:
            days.append(current_day)
            current_day += timedelta(days=1)
        return days

    def __getitem__(self, day_no):
        return self._range[day_no]

    def __len__(self):
        return len(self._range)


s1 = DateRangeSequence(date(2018, 1, 1), date(2018, 1, 5))
for day in s1:
    print(day)

print()
print(s1[0])
print(s1[3])
print(s1[-1])

'''
Pythons checks at a high level for two things: 
1) whether it contains one of the iterator methods __next__ or __iter__, 
and 2) whether it is a sequence and has __len__ and __getitem__.
'''

'''
Iterables take up O(n) indexing time whereas sequences take up O(1) indexing time.
Sequences take up O(n) memory space whereas iterables take up O(1) memory space.
'''