import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSchedulerEmbeddingOverLength(CustomTestCase):
    def test_over_length_embedding_request_is_aborted_before_enqueue(self):
        scheduler = MagicMock()
        recv_req = MagicMock(mm_inputs=None)
        with (
            patch.object(scheduler_module, "Req") as req_cls,
            patch.object(
                scheduler_module, "validate_input_length", return_value="too long"
            ),
            patch.object(scheduler_module, "get_serving"),
        ):
            Scheduler._handle_embedding_request(scheduler, recv_req, ExitStack())

        req = req_cls.return_value
        req.set_finish_with_abort.assert_called_once_with("too long")
        scheduler._add_request_to_queue.assert_called_once_with(req)


if __name__ == "__main__":
    unittest.main()
