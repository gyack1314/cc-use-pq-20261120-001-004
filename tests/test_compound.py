import copy
import json
import tempfile
import unittest
from pathlib import Path

from climate_ledger.application import Application
from climate_ledger.contracts import DomainError


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def metric(metric_id, dimension, unit, threshold, direction, minimum, maximum):
    return {"id": metric_id, "revision": 1, "name": metric_id, "dimension": dimension,
            "unit": unit, "threshold": threshold, "direction": direction,
            "minimum": minimum, "maximum": maximum}


def reading(station, metric_id, day, value, unit, quality="valid"):
    return {"station": station, "metric": metric_id, "day": day, "revision": 1,
            "value": value, "unit": unit, "quality": quality}


def missing_reading(station, metric_id, day, unit="C"):
    return reading(station, metric_id, day, None, unit, quality="missing")


class CompoundTestBase(unittest.TestCase):
    """单区单站、老年20人+成年80人的8天合成窗口，热阈值30C、缺水阈值80L。"""

    zone_id = "z1"
    station_id = "z1-s"
    start = "2026-07-01"
    end = "2026-07-08"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.app = Application(Path(self.temp.name) / "compound.sqlite3")
        self.addCleanup(self.app.ledger.close)

    def build_bundle(self, heat_values, water_values, water_unit="L"):
        """heat_values/water_values 为 (值, 质量) 序列，None 值用 missing 表示。"""
        records = [
            {"kind": "zone", "data": {"id": self.zone_id, "revision": 1, "name": "复合测试区"}},
            {"kind": "metric", "data": metric("heat", "temperature", "C", "30", "high", "-50", "60")},
            {"kind": "metric", "data": metric("water", "water", "L", "80", "low", "0", "1000")},
            {"kind": "station", "data": {"id": self.station_id, "revision": 1, "zone": self.zone_id,
                                         "start": self.start, "end": self.end, "weight": "1",
                                         "metrics": ["heat", "water"]}},
            {"kind": "population", "data": {"id": "z1-senior", "revision": 1, "zone": self.zone_id,
                                            "group": "senior", "start": self.start, "end": self.end,
                                            "count": 20, "vulnerability": "5"}},
            {"kind": "population", "data": {"id": "z1-adult", "revision": 1, "zone": self.zone_id,
                                            "group": "adult", "start": self.start, "end": self.end,
                                            "count": 80, "vulnerability": "1"}},
        ]
        for index, (heat, water) in enumerate(zip(heat_values, water_values), 1):
            day = f"2026-07-0{index}"
            records.append({"kind": "reading", "data": heat if isinstance(heat, dict)
                            else reading(self.station_id, "heat", day, str(heat), "C")})
            records.append({"kind": "reading", "data": water if isinstance(water, dict)
                            else reading(self.station_id, "water", day, str(water), water_unit)})
        self.app.ingest({"id": "compound-base", "source": "复合合成资料", "records": records})

    def base_query(self, **overrides):
        query = {"heat_metric": "heat", "water_metric": "water", "zones": [self.zone_id],
                 "start": self.start, "end": self.end, "minimum_coverage": "0.8"}
        query.update(overrides)
        return query

    @staticmethod
    def joint(day, group, heat_affected, water_affected, intersection=None):
        return {"zone": CompoundTestBase.zone_id, "day": day, "group": group,
                "heat_affected": heat_affected, "water_affected": water_affected,
                "intersection": intersection}

    def daily_map(self, result):
        return {(row["zone"], row["day"]): row for row in result["daily"]}


class HeatEpisodeRecoveryTests(CompoundTestBase):
    # 热：H H S S H H H H；水：均安全（100L）
    HEAT = [32, 32, 25, 25, 32, 32, 32, 32]
    WATER = [100] * 8

    def setUp(self):
        super().setUp()
        self.build_bundle(self.HEAT, self.WATER)

    def test_heat_event_ends_with_two_day_recovery_window(self):
        result = self.app.compound(self.base_query())
        phases = [row["heat_phase"] for row in result["daily"]]
        self.assertEqual(phases, ["heat_event", "heat_event", "recovery", "recovery",
                                  "heat_event", "heat_event", "heat_event", "heat_event"])
        episodes = result["heat_episodes"]
        self.assertEqual(len(episodes), 2)
        first, second = episodes
        self.assertEqual((first["start"], first["end"], first["hazard_days"]),
                         ("2026-07-01", "2026-07-04", 2))
        self.assertEqual(first["recovery_days"], ["2026-07-03", "2026-07-04"])
        self.assertEqual(first["right_boundary"], "observed")
        self.assertEqual(second["start"], "2026-07-05")
        self.assertEqual(second["right_boundary"], "window")

    def test_water_shortage_in_recovery_is_compound_exposure(self):
        # 把 7月3日（恢复期第一日）改为缺水
        correction = reading(self.station_id, "water", "2026-07-03", "50", "L")
        correction["revision"] = 2
        self.app.ingest({"id": "water-drought", "source": "复合合成资料", "records": [
            {"kind": "reading", "data": correction}]})
        joint = []
        for index in range(1, 9):
            day = f"2026-07-0{index}"
            heat_on = self.HEAT[index - 1] >= 30
            water_on = day == "2026-07-03"
            for group, count in (("senior", 20), ("adult", 80)):
                joint.append(self.joint(
                    day, group,
                    count if heat_on else 0,
                    count if water_on else 0,
                    count if heat_on and water_on else 0))
        result = self.app.compound(self.base_query(joint_population=joint))
        row = self.daily_map(result)[(self.zone_id, "2026-07-03")]
        self.assertEqual(row["heat_phase"], "recovery")
        self.assertEqual(row["compound_qualified"], True)
        self.assertEqual(row["compound_reason"], "recovery_water")
        # 恢复期缺水但当日无热危险：分层落在 water_only，仍计入复合口径
        self.assertEqual(row["strata"], {"both": 0, "heat_only": 0, "water_only": 100})
        events = result["compound_events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["confirmed_days"], 1)
        self.assertEqual(result["joint_totals"]["compound_person_days"]["confirmed"], 100)
        self.assertEqual(result["joint_totals"]["compound_person_days"]["recovery_water_confirmed"], 100)

    def test_single_indicator_results_remain_available_and_unchanged(self):
        result = self.app.compound(self.base_query())
        heat_single = self.app.analyze({"metric": "heat", "zones": [self.zone_id],
                                        "start": self.start, "end": self.end})
        self.assertEqual(result["single_indicator"]["heat"]["total"], heat_single["total"])
        self.assertIn("water", result["single_indicator"])
        # 旧单指标查询依然可用
        self.assertEqual(self.app.analyze({"metric": "water", "zones": [self.zone_id],
                                           "start": self.start, "end": self.end})["status"], "qualified")

    def test_drought_across_event_and_recovery_merges_into_one_compound_event(self):
        # 热 H H S S ...；缺水落在 7月2日（事件内）与 7月3日（恢复期），应合并为一个复合事件
        for day, value in (("2026-07-02", "50"), ("2026-07-03", "50")):
            correction = reading(self.station_id, "water", day, value, "L")
            correction["revision"] = 2
            self.app.ingest({"id": f"water-{day}", "source": "复合合成资料",
                             "records": [{"kind": "reading", "data": correction}]})
        drought_days = {"2026-07-02", "2026-07-03"}
        joint = []
        for index in range(1, 9):
            day = f"2026-07-0{index}"
            heat_on = self.HEAT[index - 1] >= 30
            water_on = day in drought_days
            for group, count in (("senior", 20), ("adult", 80)):
                joint.append(self.joint(day, group, count if heat_on else 0,
                                        count if water_on else 0,
                                        count if heat_on and water_on else 0))
        result = self.app.compound(self.base_query(joint_population=joint))
        events = result["compound_events"]
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0]["start"], events[0]["end"]), ("2026-07-02", "2026-07-03"))
        self.assertEqual(events[0]["confirmed_days"], 2)
        # 事件依据保留每日热相位与双指标读数
        phases = {d["day"]: d["heat_phase"] for d in events[0]["days_detail"]}
        self.assertEqual(phases, {"2026-07-02": "heat_event", "2026-07-03": "recovery"})
        # 事件内 both=100，恢复期 water_only=100，复合口径合计200
        self.assertEqual(result["joint_totals"]["compound_person_days"]["confirmed"], 200)


class MissingBoundaryTests(CompoundTestBase):
    # 热：H H 缺测 S H H H H（S为安全日）；水：均安全
    HEAT = [32, 32, None, 25, 32, 32, 32, 32]
    WATER = [100] * 8

    def setUp(self):
        super().setUp()
        heat = [missing_reading(self.station_id, "heat", f"2026-07-0{i}") if value is None else value
                for i, value in enumerate(self.HEAT, 1)]
        self.build_bundle(heat, self.WATER)

    def test_missing_day_neither_ends_nor_continues_heat_event(self):
        result = self.app.compound(self.base_query())
        phases = [row["heat_phase"] for row in result["daily"]]
        self.assertEqual(phases, ["heat_event", "heat_event", "heat_unknown",
                                  "heat_safe", "heat_event", "heat_event", "heat_event", "heat_event"])
        episodes = result["heat_episodes"]
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]["right_boundary"], "unknown")
        self.assertEqual(episodes[1]["left_boundary"], "possible_link")
        self.assertEqual(episodes[1]["connection"], "possible")
        # 缺测区日在汇总中保持未知，不被当作安全日
        self.assertEqual(result["joint_totals"]["unknown_zone_days"], 1)
        self.assertIn("heat_unknown", result["joint_totals"]["uncertainty_reasons"])

    def test_drought_on_unknown_heat_day_is_only_possibly_compound(self):
        correction = reading(self.station_id, "water", "2026-07-03", "40", "L")
        correction["revision"] = 2
        self.app.ingest({"id": "drought-gap", "source": "复合合成资料", "records": [
            {"kind": "reading", "data": correction}]})
        result = self.app.compound(self.base_query())
        row = self.daily_map(result)[(self.zone_id, "2026-07-03")]
        self.assertIsNone(row["compound_qualified"])
        self.assertEqual(row["compound_status"], "possible")
        event = result["compound_events"][0]
        self.assertEqual(event["possible_days"], 1)
        self.assertTrue(event["boundary_unknown"])

    def test_missing_water_day_does_not_invent_compound_event(self):
        # 热缺测且水也缺测：不能凭未知日构造复合事件
        water_gap = missing_reading(self.station_id, "water", "2026-07-03", unit="L")
        water_gap["revision"] = 2
        self.app.ingest({"id": "water-gap", "source": "复合合成资料", "records": [
            {"kind": "reading", "data": water_gap}]})
        result = self.app.compound(self.base_query())
        row = self.daily_map(result)[(self.zone_id, "2026-07-03")]
        self.assertIsNone(row["compound_qualified"])
        self.assertEqual(row["compound_status"], "unknown")
        self.assertEqual(result["compound_events"], [])
        self.assertIn("water_unknown", row["unknown_reasons"])

    def test_corrected_reading_reconnects_episodes_under_same_evidence(self):
        before = self.app.compound(self.base_query())
        self.assertEqual(len(before["heat_episodes"]), 2)
        # 更正缺测日为安全日：前段获得 observed 边界，恢复资格重新计算
        self.app.ingest({"id": "heat-fix", "source": "复合合成资料", "records": [
            {"kind": "reading", "data": {"station": self.station_id, "metric": "heat",
                                         "day": "2026-07-03", "revision": 2,
                                         "value": "25", "unit": "C", "quality": "valid"}}]})
        after = self.app.compound(self.base_query())
        phases = [row["heat_phase"] for row in after["daily"]]
        self.assertEqual(phases, ["heat_event", "heat_event", "recovery", "recovery",
                                  "heat_event", "heat_event", "heat_event", "heat_event"])
        self.assertEqual(len(after["heat_episodes"]), 2)
        self.assertEqual(after["heat_episodes"][0]["right_boundary"], "observed")
        self.assertEqual(after["heat_episodes"][0]["recovery_days"], ["2026-07-03", "2026-07-04"])
        self.assertEqual(after["heat_episodes"][1]["left_boundary"], "window")
        self.assertEqual(after["heat_episodes"][1]["connection"], "confirmed")
        # 旧水位结果仍可复现：缺测版本仍是 unknown / possible_link
        old = self.app.compound(self.base_query(), seq=1)
        self.assertEqual(old["heat_episodes"], before["heat_episodes"])


class MutuallyExclusiveStrataTests(CompoundTestBase):
    # 前两天双危险，第三天仅热，第四天仅水，其余安全
    HEAT = [32, 32, 32, 25, 25, 25, 25, 25]
    WATER = [50, 50, 100, 50, 100, 100, 100, 100]

    def setUp(self):
        super().setUp()
        self.build_bundle(self.HEAT, self.WATER)

    def full_joint(self, intersection_override=None):
        joint = []
        for index in range(1, 9):
            day = f"2026-07-0{index}"
            heat_on = self.HEAT[index - 1] >= 30
            water_on = self.WATER[index - 1] <= 80
            for group, count in (("senior", 20), ("adult", 80)):
                h = count if heat_on else 0
                w = count if water_on else 0
                intersection = intersection_override if intersection_override is not None else (
                    min(h, w))
                joint.append(self.joint(day, group, h, w, intersection))
        return joint

    def test_three_strata_partition_population_and_reconcile(self):
        result = self.app.compound(self.base_query(joint_population=self.full_joint()))
        day1 = self.daily_map(result)[(self.zone_id, "2026-07-01")]
        self.assertEqual(day1["strata"], {"both": 100, "heat_only": 0, "water_only": 0})
        day3 = self.daily_map(result)[(self.zone_id, "2026-07-03")]
        self.assertEqual(day3["strata"], {"both": 0, "heat_only": 100, "water_only": 0})
        day4 = self.daily_map(result)[(self.zone_id, "2026-07-04")]
        self.assertEqual(day4["strata"], {"both": 0, "heat_only": 0, "water_only": 100})
        # 互斥分层相加等于当日进入联合分母的人数（安全日无人受影响，即为0）
        for row in result["daily"]:
            if row["strata"] is None:
                continue
            affected = sum(
                (g["joint_input"]["heat_affected"] + g["joint_input"]["water_affected"]
                 - (g["joint_input"]["intersection"] or 0))
                for g in row["groups"] if g["joint_input"])
            self.assertEqual(sum(row["strata"].values()), affected)
            self.assertLessEqual(sum(row["strata"].values()), row["population"]["population"])
        totals = result["joint_totals"]
        self.assertEqual(totals["confirmed"]["both"], 200)
        self.assertEqual(totals["confirmed"]["heat_only"], 100)
        self.assertEqual(totals["confirmed"]["water_only"], 100)
        self.assertEqual(totals["heat_affected_person_days"]["confirmed"], 300)
        self.assertEqual(totals["water_affected_person_days"]["confirmed"], 300)

    def test_incompatible_joint_counts_rejected(self):
        joint = self.full_joint()
        joint[0]["heat_affected"] = 150  # 超过老年组20人分母
        with self.assertRaises(DomainError) as error:
            self.app.compound(self.base_query(joint_population=joint))
        self.assertEqual(error.exception.code, "joint_population")

    def test_incompatible_intersection_rejected(self):
        joint = self.full_joint(intersection_override=5)
        # 7月1日两组都是全部受热受水，交集必须等于100%/组人数
        with self.assertRaises(DomainError) as error:
            self.app.compound(self.base_query(joint_population=joint))
        self.assertEqual(error.exception.code, "joint_population")

    def test_safe_day_with_affected_counts_rejected(self):
        joint = self.full_joint()
        joint[6]["heat_affected"] = 10  # 7月4日热安全却登记受热
        with self.assertRaises(DomainError) as error:
            self.app.compound(self.base_query(joint_population=joint))
        self.assertEqual(error.exception.code, "joint_population")

    def test_unknown_intersection_yields_frechet_bounds_not_point_estimate(self):
        joint = self.full_joint()
        # 老年组7月1日：15受热、12缺水、交集未知（题述示例口径，组内20人）
        senior_day1 = next(r for r in joint if r["day"] == "2026-07-01" and r["group"] == "senior")
        senior_day1.update(heat_affected=15, water_affected=12, intersection=None)
        result = self.app.compound(self.base_query(joint_population=joint))
        group = next(g for g in self.daily_map(result)[(self.zone_id, "2026-07-01")]["groups"]
                     if g["group"] == "senior")
        self.assertFalse(group["intersection_known"])
        self.assertEqual(group["strata"]["both"], {"lower": 7, "upper": 12})
        self.assertEqual(group["strata"]["heat_only"], {"lower": 3, "upper": 8})
        self.assertEqual(group["strata"]["water_only"], {"lower": 0, "upper": 5})
        self.assertIn("intersection_unknown", group["uncertainty"])
        # 汇总只给区间，不产生确定保护数
        totals = result["joint_totals"]
        self.assertGreater(totals["bounded"]["both"]["upper"], totals["bounded"]["both"]["lower"])
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("intersection_unknown", totals["uncertainty_reasons"])

    def test_missing_joint_population_leaves_strata_unknown(self):
        joint = [r for r in self.full_joint() if r["day"] != "2026-07-01"]
        result = self.app.compound(self.base_query(joint_population=joint))
        row = self.daily_map(result)[(self.zone_id, "2026-07-01")]
        self.assertIsNone(row["strata"])
        self.assertIn("joint_population_gap", row["unknown_reasons"])


class SharedFacilityAllocationTests(CompoundTestBase):
    # 连续4天双危险，足够展示名额共享
    HEAT = [32, 32, 32, 32, 25, 25, 25, 25]
    WATER = [50, 50, 50, 50, 100, 100, 100, 100]

    def setUp(self):
        super().setUp()
        self.build_bundle(self.HEAT, self.WATER)
        joint = []
        for index in range(1, 9):
            day = f"2026-07-0{index}"
            heat_on = self.HEAT[index - 1] >= 30
            water_on = self.WATER[index - 1] <= 80
            for group, count in (("senior", 20), ("adult", 80)):
                h = count if heat_on else 0
                w = count if water_on else 0
                joint.append(self.joint(day, group, h, w, min(h, w)))
        self.joint = joint

    def facility(self, capacity=10):
        return [{"facility": "shared-hub", "zone": self.zone_id, "daily_capacity": capacity,
                 "services": [
                     {"service": "joint-relief", "hazards": ["heat", "water"], "cost_per_person": 2,
                      "groups": ["senior", "adult"], "priority": 30},
                     {"service": "cooling", "hazards": ["heat"], "cost_per_person": 1,
                      "groups": ["senior", "adult"], "priority": 20},
                     {"service": "water-supply", "hazards": ["water"], "cost_per_person": 1,
                      "groups": ["senior", "adult"], "priority": 10}]}]

    def test_joint_service_costs_two_slots_and_person_counted_once(self):
        result = self.app.compound(self.base_query(joint_population=self.joint,
                                                   facilities=self.facility()))
        day = next(d for d in result["joint_actions"]["days"] if d["day"] == "2026-07-01")
        allocation = day["allocation"]
        # 10个名额，联合服务一人2名额 => 最多服务5名老年（脆弱优先），且只计一次保护
        self.assertEqual(allocation["protected_total"], 5)
        self.assertEqual(allocation["protected_by_stratum"]["both"], 5)
        joint_log = next(s for s in allocation["services"] if s["service"] == "joint-relief")
        self.assertEqual(joint_log["used_slots"], 10)
        self.assertEqual(joint_log["assignments"], [{"group": "senior", "stratum": "both", "count": 5}])
        # 名额被联合服务耗尽，单项服务不能再全额使用同一设施
        for name in ("cooling", "water-supply"):
            self.assertEqual(next(s for s in allocation["services"] if s["service"] == name)["used_slots"], 0)
        # 各服务槽位消耗之和恰好等于设施总容量
        self.assertEqual(sum(s["used_slots"] for s in allocation["services"]), 10)

    def test_single_hazard_services_share_capacity_without_double_count(self):
        # 仅保留降温与供水（各占1名额），容量10
        facility = [{"facility": "shared-hub", "zone": self.zone_id, "daily_capacity": 10,
                     "services": [
                         {"service": "cooling", "hazards": ["heat"], "cost_per_person": 1,
                          "groups": ["senior", "adult"], "priority": 20},
                         {"service": "water-supply", "hazards": ["water"], "cost_per_person": 1,
                          "groups": ["senior", "adult"], "priority": 10}]}]
        result = self.app.compound(self.base_query(joint_population=self.joint, facilities=facility))
        day = next(d for d in result["joint_actions"]["days"] if d["day"] == "2026-07-01")
        allocation = day["allocation"]
        # 双危险日所有人都在 both 分层：先到的降温服务10名老人，供水无法重复服务同一批人
        self.assertEqual(allocation["protected_total"], 10)
        cooling = next(s for s in allocation["services"] if s["service"] == "cooling")
        supply = next(s for s in allocation["services"] if s["service"] == "water-supply")
        self.assertEqual(cooling["used_slots"], 10)
        self.assertEqual(supply["used_slots"], 0)
        self.assertEqual(supply["assignments"], [])
        # 设施总名额被降温耗尽，各服务槽位消耗之和不超过总容量
        self.assertEqual(sum(s["used_slots"] for s in allocation["services"]), 10)
        self.assertEqual(sum(allocation["protected_by_stratum"].values()), 10)

    def test_multiple_facilities_keep_independent_capacity_but_person_served_once(self):
        facilities = [
            {"facility": "hub-a", "zone": self.zone_id, "daily_capacity": 6, "services": [
                {"service": "cooling", "hazards": ["heat"], "cost_per_person": 1,
                 "groups": ["senior", "adult"], "priority": 20}]},
            {"facility": "hub-b", "zone": self.zone_id, "daily_capacity": 4, "services": [
                {"service": "water-supply", "hazards": ["water"], "cost_per_person": 1,
                 "groups": ["senior", "adult"], "priority": 10}]}]
        result = self.app.compound(self.base_query(joint_population=self.joint, facilities=facilities))
        day = next(d for d in result["joint_actions"]["days"] if d["day"] == "2026-07-01")["allocation"]
        # 两个设施各自满负荷（6+4），但服务的是不同老人，同一人不重复计数
        self.assertEqual(day["protected_total"], 10)
        self.assertEqual(sum(day["facility_remaining"].values()), 0)
        cooling = next(s for s in day["services"] if s["service"] == "cooling")
        supply = next(s for s in day["services"] if s["service"] == "water-supply")
        self.assertEqual(cooling["assignments"][0]["count"], 6)
        self.assertEqual(supply["assignments"][0]["count"], 4)

    def test_allocation_unknown_intersection_reports_bounds_only(self):
        joint = copy.deepcopy(self.joint)
        senior = next(r for r in joint if r["day"] == "2026-07-01" and r["group"] == "senior")
        senior.update(heat_affected=15, water_affected=12, intersection=None)
        result = self.app.compound(self.base_query(joint_population=joint, facilities=self.facility()))
        day = next(d for d in result["joint_actions"]["days"] if d["day"] == "2026-07-01")
        self.assertFalse(day["intersection_known"])
        self.assertIn("intersection_unknown", day["uncertainty"])
        lower, upper = day["bounds"]["lower"], day["bounds"]["upper"]
        self.assertLessEqual(lower["protected_total"], upper["protected_total"])
        self.assertIn("note", day)


class CompoundPlanTests(CompoundTestBase):
    HEAT = [32, 32, 32, 32, 25, 25, 25, 25]
    WATER = [50, 50, 50, 50, 100, 100, 100, 100]

    def setUp(self):
        super().setUp()
        self.build_bundle(self.HEAT, self.WATER)
        joint = []
        for index in range(1, 9):
            day = f"2026-07-0{index}"
            heat_on = self.HEAT[index - 1] >= 30
            water_on = self.WATER[index - 1] <= 80
            for group, count in (("senior", 20), ("adult", 80)):
                h = count if heat_on else 0
                w = count if water_on else 0
                joint.append(self.joint(day, group, h, w, min(h, w)))
        self.joint = joint

    def package_query(self, facilities, packages, budget=100000):
        return {"query": self.base_query(joint_population=self.joint, facilities=facilities),
                "budget_cents": budget, "packages": packages}

    def test_packages_evaluated_on_joint_feasibility_and_marked_hypothetical(self):
        facilities = [
            {"facility": "small", "zone": self.zone_id, "daily_capacity": 10, "services": [
                {"service": "cooling", "hazards": ["heat"], "cost_per_person": 1,
                 "groups": ["senior", "adult"], "priority": 20}]},
            {"facility": "large", "zone": self.zone_id, "daily_capacity": 40, "services": [
                {"service": "cooling", "hazards": ["heat"], "cost_per_person": 1,
                 "groups": ["senior", "adult"], "priority": 20}]}]
        packages = [{"package": "small", "cost_cents": 3000},
                    {"package": "large", "cost_cents": 9000}]
        result = self.app.compound_plan(self.package_query(facilities, packages))
        self.assertEqual(result["status"], "hypothetical")
        self.assertEqual(result["selected"], "large")
        values = {e["package"]: e["protected_lower"] for e in result["evaluations"]}
        self.assertEqual(values, {"small": 40, "large": 160})
        # 假设组合不写入总账
        self.assertEqual(self.app.inspect()["objects"]["action"], 0)

    def test_budget_exclusion_and_undeclared_facility(self):
        facilities = [{"facility": "small", "zone": self.zone_id, "daily_capacity": 10, "services": [
            {"service": "cooling", "hazards": ["heat"], "cost_per_person": 1,
             "groups": ["senior", "adult"], "priority": 20}]}]
        packages = [{"package": "small", "cost_cents": 5000},
                    {"package": "ghost", "cost_cents": 1000}]
        result = self.app.compound_plan(self.package_query(facilities, packages, budget=4000))
        reasons = {e["package"]: e["reason"] for e in result["excluded"]}
        self.assertEqual(reasons, {"small": "over_budget", "ghost": "facility_not_declared"})
        self.assertIsNone(result["selected"])

    def test_unknown_intersection_keeps_ranking_inconclusive(self):
        joint = copy.deepcopy(self.joint)
        # 老年组交集未知（20人中15受热、12缺水，交集可行域[7,12]）
        for day in (f"2026-07-0{i}" for i in range(1, 5)):
            senior = next(r for r in joint if r["day"] == day and r["group"] == "senior")
            senior.update(heat_affected=15, water_affected=12, intersection=None)
        # 联合服务只面向老年组 both 分层；容量（人数）覆盖交集可行域[7,12]之外，
        # 上界受交集假设约束，从而产生非平凡保护区间
        def hub(facility_id, people):
            return {"facility": facility_id, "zone": self.zone_id, "daily_capacity": people * 2,
                    "services": [{"service": "joint-relief", "hazards": ["heat", "water"],
                                  "cost_per_person": 2, "groups": ["senior"], "priority": 30}]}
        facilities = [hub("small", 8), hub("large", 10)]
        packages = [{"package": "small", "cost_cents": 3000},
                    {"package": "large", "cost_cents": 4000}]
        query = {"heat_metric": "heat", "water_metric": "water", "zones": [self.zone_id],
                 "start": self.start, "end": self.end, "joint_population": joint,
                 "facilities": facilities}
        result = self.app.compound_plan({"query": query, "budget_cents": 100000, "packages": packages})
        large = next(e for e in result["evaluations"] if e["package"] == "large")
        small = next(e for e in result["evaluations"] if e["package"] == "small")
        # 单日保护数随交集假设变化，区间非平凡
        first_day = next(d for d in large["days"] if "protected_lower" in d)
        self.assertLess(first_day["protected_lower"], first_day["protected_upper"])
        # 两个候选的保护区间重叠，不能给出确定排名
        self.assertTrue(any("large" in pair and "small" in pair for pair in result["ranking_inconclusive"]))


class ValidationTests(MutuallyExclusiveStrataTests):
    def test_unknown_metric_and_duplicate_metric_rejected(self):
        with self.assertRaises(DomainError):
            self.app.compound({**self.base_query(), "heat_metric": "nope"})
        with self.assertRaises(DomainError):
            self.app.compound({**self.base_query(), "water_metric": "heat"})

    def test_joint_population_unknown_group_rejected(self):
        joint = self.full_joint()
        joint[0]["group"] = "ghost"
        with self.assertRaises(DomainError) as error:
            self.app.compound(self.base_query(joint_population=joint))
        self.assertEqual(error.exception.code, "joint_population")

    def test_facility_unknown_group_or_zone_rejected(self):
        bad_group = [{"facility": "f", "zone": self.zone_id, "daily_capacity": 5, "services": [
            {"service": "s", "hazards": ["heat"], "cost_per_person": 1,
             "groups": ["ghost"], "priority": 1}]}]
        with self.assertRaises(DomainError) as error:
            self.app.compound(self.base_query(facilities=bad_group))
        self.assertEqual(error.exception.code, "facility")
        bad_zone = [{"facility": "f", "zone": "elsewhere", "daily_capacity": 5, "services": [
            {"service": "s", "hazards": ["heat"], "cost_per_person": 1,
             "groups": ["senior"], "priority": 1}]}]
        with self.assertRaises(DomainError) as error:
            self.app.compound(self.base_query(joint_population=self.full_joint(), facilities=bad_zone))
        self.assertEqual(error.exception.code, "facility")

    def test_illegal_hazard_declaration_rejected(self):
        bad_hazard = [{"facility": "f", "zone": self.zone_id, "daily_capacity": 5, "services": [
            {"service": "s", "hazards": ["quake"], "cost_per_person": 1,
             "groups": ["senior"], "priority": 1}]}]
        with self.assertRaises(DomainError) as error:
            self.app.compound(self.base_query(facilities=bad_hazard))
        self.assertEqual(error.exception.code, "hazards")


class ExampleContractTests(unittest.TestCase):
    def test_example_compound_query_runs_through_application(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = Application(Path(temporary) / "city.sqlite3")
            self.addCleanup(app.ledger.close)
            app.ingest(json.loads((EXAMPLES / "city.json").read_text(encoding="utf-8")))
            spec = json.loads((EXAMPLES / "compound.json").read_text(encoding="utf-8"))
            result = app.compound(spec)
            self.assertEqual(result["status"], "qualified")
            self.assertIn("single_indicator", result)
            self.assertIn("joint_actions", result)


if __name__ == "__main__":
    unittest.main()
