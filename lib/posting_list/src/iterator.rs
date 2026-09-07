use std::iter::FusedIterator;

use common::types::PointOffsetType;

use crate::value_handler::{PostingValue, SizedHandler};
use crate::visitor::PostingVisitor;
use crate::{CHUNK_LEN, PostingElement, SizedValue};

pub struct PostingIterator<'a, V: PostingValue> {
    visitor: PostingVisitor<'a, V>,
    current_elem: Option<PostingElement<V>>,
    offset: usize,
}

impl<'a, V> PostingIterator<'a, V>
where
    V: PostingValue<Handler = SizedHandler<V>> + SizedValue,
{
    /// Visit all remaining elements up to and including `last_id`.
    #[inline]
    pub fn for_each_till_id_sized(
        &mut self,
        last_id: PointOffsetType,
        mut f: impl FnMut(PostingElement<V>),
    ) {
        while self.offset / CHUNK_LEN < self.visitor.list.chunks_len() {
            let chunk_idx = self.offset / CHUNK_LEN;
            let start = self.offset % CHUNK_LEN;
            let (ids, values) = self.visitor.chunk_ids_and_sized_values(chunk_idx);
            let count = ids[start..].partition_point(|id| *id <= last_id);

            for (&id, &value) in
                std::iter::zip(&ids[start..start + count], &values[start..start + count])
            {
                f(PostingElement { id, value });
            }

            self.offset += count;
            self.current_elem = None;
            if start + count != CHUNK_LEN {
                if self.offset < self.visitor.len() {
                    self.current_elem = self.visitor.get_by_offset(self.offset);
                }
                return;
            }
        }

        while self.offset < self.visitor.len() {
            let remainder_idx = self.offset - self.visitor.list.chunks_len() * CHUNK_LEN;
            let Some(remainder) = self.visitor.list.get_remainder(remainder_idx) else {
                self.offset = self.visitor.len();
                self.current_elem = None;
                return;
            };
            let element = PostingElement {
                id: remainder.id.get(),
                value: remainder.value,
            };
            if element.id > last_id {
                self.current_elem = Some(element);
                return;
            }

            self.offset += 1;
            self.current_elem = None;
            f(element);
        }
    }
}

impl<'a, V: PostingValue> PostingIterator<'a, V> {
    pub fn new(visitor: PostingVisitor<'a, V>) -> Self {
        Self {
            visitor,
            current_elem: None,
            offset: 0,
        }
    }

    /// Advances the iterator until the current element id is greater than or equal to the given id.
    ///
    /// Returns `Some(PostingElement)` on the first element that is greater than or equal to the given id. It can be possible that this id is
    /// the head of the iterator, so it does not need to be advanced.
    ///
    /// `None` means the iterator is exhausted.
    pub fn advance_until_greater_or_equal(
        &mut self,
        target_id: PointOffsetType,
    ) -> Option<PostingElement<V>> {
        if let Some(current) = &self.current_elem
            && current.id >= target_id
        {
            return Some(current.clone());
        }

        if self.offset >= self.visitor.len() {
            return None;
        }

        let Some(offset) = self
            .visitor
            .search_greater_or_equal(target_id, Some(self.offset))
        else {
            self.current_elem = None;
            self.offset = self.visitor.len();
            return None;
        };

        debug_assert!(offset >= self.offset);
        let greater_or_equal = self.visitor.get_by_offset(offset);

        self.current_elem = greater_or_equal.clone();
        self.offset = offset;

        greater_or_equal
    }

    /// Visit all remaining elements up to and including `last_id`.
    #[inline]
    pub fn for_each_till_id(
        &mut self,
        last_id: PointOffsetType,
        mut f: impl FnMut(PostingElement<V>),
    ) {
        while self.offset < self.visitor.len() {
            let Some(element) = self.visitor.get_by_offset(self.offset) else {
                self.current_elem = None;
                self.offset = self.visitor.len();
                break;
            };
            if element.id > last_id {
                self.current_elem = Some(element);
                break;
            }

            self.offset += 1;
            self.current_elem = None;
            f(element);
        }
    }
}

impl<V: PostingValue> Iterator for PostingIterator<'_, V> {
    type Item = PostingElement<V>;

    fn next(&mut self) -> Option<Self::Item> {
        let next_opt = self.visitor.get_by_offset(self.offset).inspect(|_| {
            self.offset += 1;
        });

        self.current_elem = next_opt.clone();

        next_opt
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_len = self.len();
        (remaining_len, Some(remaining_len))
    }

    fn count(self) -> usize {
        self.size_hint().0
    }
}

impl<V: PostingValue> ExactSizeIterator for PostingIterator<'_, V> {
    fn len(&self) -> usize {
        self.visitor.list.len().saturating_sub(self.offset)
    }
}

impl<V: PostingValue> FusedIterator for PostingIterator<'_, V> {}

#[cfg(test)]
mod tests {
    use crate::PostingList;

    fn collect_sized_until(
        iterator: &mut super::PostingIterator<'_, u32>,
        last_id: u32,
    ) -> Vec<(u32, u32)> {
        let mut visited = Vec::new();
        iterator.for_each_till_id_sized(last_id, |element| {
            visited.push((element.id, element.value));
        });
        visited
    }

    #[test]
    fn sized_batch_iteration_preserves_cursor_across_chunks() {
        let posting = (0..300u32)
            .map(|value| (value * 2, value))
            .collect::<PostingList<u32>>();
        let mut iterator = posting.iter();
        let mut actual = Vec::new();

        iterator.for_each_till_id_sized(137, |element| {
            actual.push((element.id, element.value));
        });
        iterator.for_each_till_id_sized(401, |element| {
            actual.push((element.id, element.value));
        });
        iterator.for_each_till_id_sized(u32::MAX, |element| {
            actual.push((element.id, element.value));
        });

        let expected = (0..300u32)
            .map(|value| (value * 2, value))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert_eq!(iterator.len(), 0);
    }

    #[test]
    fn sized_batch_iteration_stops_before_the_first_larger_id() {
        let posting = (0..300u32)
            .map(|value| (value * 2, value))
            .collect::<PostingList<u32>>();
        let mut iterator = posting.iter();

        assert_eq!(collect_sized_until(&mut iterator, 0), vec![(0, 0)]);
        assert_eq!(iterator.next().map(|element| element.id), Some(2));
        assert_eq!(collect_sized_until(&mut iterator, 4), vec![(4, 2)]);
        assert_eq!(iterator.next().map(|element| element.id), Some(6));

        let element = iterator.advance_until_greater_or_equal(254).unwrap();
        assert_eq!(element.id, 254);
        assert!(collect_sized_until(&mut iterator, 253).is_empty());
        assert_eq!(
            collect_sized_until(&mut iterator, 258),
            vec![(254, 127), (256, 128), (258, 129)]
        );
        assert_eq!(iterator.next().map(|element| element.id), Some(260));
    }

    #[test]
    fn generic_batch_iteration_preserves_the_cursor() {
        let posting = (0..140u32)
            .map(|value| (value * 3, ()))
            .collect::<PostingList<()>>();
        let mut iterator = posting.iter();
        let mut visited = Vec::new();

        iterator.for_each_till_id(191, |element| visited.push(element.id));
        assert_eq!(visited, (0..64).map(|value| value * 3).collect::<Vec<_>>());
        assert_eq!(iterator.next().map(|element| element.id), Some(192));

        iterator.for_each_till_id(u32::MAX, |element| visited.push(element.id));
        assert_eq!(visited.len(), 139);
        assert_eq!(visited.last(), Some(&417));
        assert_eq!(iterator.len(), 0);
    }
}
